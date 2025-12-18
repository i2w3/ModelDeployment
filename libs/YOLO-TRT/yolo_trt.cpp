#include "yolo_trt.h"

YOLODetector::YOLODetector(const std::string& modelPath, const ModelParams& modelParams_) 
    :net(modelPath), modelParams(modelParams_) {
    std::cout << "Initialized YOLODetector with model path: " << modelPath << std::endl;
}


std::vector<YOLODetection> YOLODetector::infer(const cv::Mat& image) {
#ifdef IS_DEBUG
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double, std::milli> preprocess_time;
    start = std::chrono::high_resolution_clock::now();
#endif
    // 前处理
    PreprocessParams params = this->preprocess(image);
    std::unordered_map<std::string, cv::Mat> input_blobs = {{"images", params.blob}};
#ifdef IS_DEBUG
    end = std::chrono::high_resolution_clock::now();
    preprocess_time = end - start;
    std::cout << "Preprocessing time: " << preprocess_time.count() << " ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
#endif
    // 正向传播
    std::unordered_map<std::string, cv::Mat> outputs = this->net(input_blobs);
#ifdef IS_DEBUG
    end = std::chrono::high_resolution_clock::now();
    preprocess_time = end - start;
    std::cout << "model forward time: " << preprocess_time.count() << " ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
#endif
    // 后处理
    auto result = this->postprocess(outputs, params);
#ifdef IS_DEBUG
    end = std::chrono::high_resolution_clock::now();
    preprocess_time = end - start;
    std::cout << "Postprocessing time: " << preprocess_time.count() << " ms" << std::endl;
#endif
    return result;
}


PreprocessParams YOLODetector::preprocess(const cv::Mat& image) {
    int image_h = image.rows;
    int image_w = image.cols;

    float scale = std::min(static_cast<float>(this->modelParams.image_size) / image_w, 
                           static_cast<float>(this->modelParams.image_size) / image_h);
    int resized_w = static_cast<int>(image_w * scale);
    int resized_h = static_cast<int>(image_h * scale);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h));

    int pad_w = this->modelParams.image_size  - resized_w;
    int pad_h = this->modelParams.image_size - resized_h;

    int top = static_cast<int>(pad_h - 0.1);
    int bottom = static_cast<int>(pad_h + 0.1);
    int left = static_cast<int>(pad_w - 0.1);
    int right = static_cast<int>(pad_w + 0.1);

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat blob = cv::dnn::blobFromImage(padded, 
                                          1.0 / 255.0, 
                                          cv::Size(this->modelParams.image_size, this->modelParams.image_size), 
                                          cv::Scalar(), 
                                          true, 
                                          false);
    return PreprocessParams(image, blob, top, bottom, left, right, scale);
}


std::vector<YOLODetection> YOLODetector::postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) {
    std::vector<YOLODetection> results;
    if(outputs.size() == 1) {
        if (outputs.at("output0").dims == 2) {
            throw std::runtime_error("Unsupported number of output blobs.");
        }
        else if (outputs.at("output0").dims == 3) {
            if (this->modelParams.num_classes + 4 == outputs.at("output0").size[1]) {
                // det 模型
                // std::cout << "Post-processing detection results..." << std::endl;
                results = this->postprocessDetection({outputs.at("output0")}, params);
            }
            else {
                throw std::runtime_error("Unsupported output dimensions.");
            }
            
        }
        else {
            throw std::runtime_error("Unsupported output dimensions.");
        }
    }
    else {
        throw std::runtime_error("Unsupported number of output blobs.");
    }
    return results;
}


std::vector<YOLODetection> YOLODetector::postprocessDetection(const std::vector<cv::Mat>& outputs, const PreprocessParams& params) {
    std::vector<YOLODetection> detections;
    auto output = outputs[0];

    // 获取输出维度: [1, num_classes + 4, num_boxes]
    // 4表示: center_x, center_y, width, height
    const int num_boxes = output.size[2];        // 检测框总数

    // 重塑并转置输出矩阵为 [num_boxes, num_classes + 4] 格式
    // 这样可以更容易地遍历每个检测框
    cv::Mat output_mat = output.reshape(1, output.size[1]);
    output_mat = output_mat.t();

    // 预分配vector以避免频繁的内存重新分配
    // 根据最大可能的检测数预留空间 (所有框都可能通过阈值)
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    boxes.reserve(num_boxes);
    confidences.reserve(num_boxes);
    class_ids.reserve(num_boxes);

    // 处理每个检测框
    for (int i = 0; i < num_boxes; i++) {
        cv::Mat row = output_mat.row(i);
        float* data = row.ptr<float>();

        // 提取边界框坐标 (中心格式: xc, yc, w, h)
        float xc = data[0];  // 中心x坐标
        float yc = data[1];  // 中心y坐标
        float w = data[2];   // 宽度
        float h = data[3];   // 高度

        // 提取类别置信度分数 (此检测的所有类别)
        cv::Mat scores = row.colRange(4, 4 + this->modelParams.num_classes);

        // 找到置信度最高的类别
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id_point);

        // 过滤掉低于置信度阈值的检测
        if (max_class_score > this->modelParams.conf_threshold) {
            // 从中心坐标转换为左上角坐标
            float x = xc - w / 2.0f;  // 左边界
            float y = yc - h / 2.0f;  // 上边界

            // 将检测结果存储在临时vector中以供NMS处理
            boxes.emplace_back(cv::Rect(static_cast<int>(x),
                                        static_cast<int>(y),
                                        static_cast<int>(w),
                                        static_cast<int>(h)));
            confidences.emplace_back(static_cast<float>(max_class_score));
            class_ids.emplace_back(class_id_point.x);  // class_id_point.x 包含类别索引
        }
    }

    // 应用非极大值抑制(NMS)消除冗余的重叠框
    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confidences, 0.0f, this->modelParams.nms_threshold, indices);
    }

    // 将过滤后的索引转换为最终的YOLODetection对象
    detections.reserve(indices.size());  // 预分配最终结果vector
    for (int idx : indices) {
        YOLODetection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }
    for (auto& det : detections) {
        // 调整边界框以补偿前处理中的缩放和填充
        float x_scale = static_cast<float>(params.original.cols) / (this->modelParams.image_size - params.left - params.right);
        float y_scale = static_cast<float>(params.original.rows) / (this->modelParams.image_size - params.top - params.bottom);

        det.box.x = static_cast<int>((det.box.x - params.left) * x_scale);
        det.box.y = static_cast<int>((det.box.y - params.top) * y_scale);
        det.box.width = static_cast<int>(det.box.width * x_scale);
        det.box.height = static_cast<int>(det.box.height * y_scale);
    }
    return detections;
}