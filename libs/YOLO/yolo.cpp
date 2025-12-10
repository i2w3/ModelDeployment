#include "yolo.h"

YOLODetector::YOLODetector(const std::string& modelPath, const ModelParams& modelParams_) 
    :modelParams(modelParams_) {
#ifdef IS_DEBUG
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double, std::milli> preprocess_time;
    start = std::chrono::high_resolution_clock::now();
#endif
    std::cout << "Initialized YOLODetector with model path: " << modelPath << std::endl;
    this->net = cv::dnn::readNetFromONNX(modelPath);
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#ifdef IS_DEBUG
    end = std::chrono::high_resolution_clock::now();
    preprocess_time = end - start;
    std::cout << "Model loading time: " << preprocess_time.count() << " ms" << std::endl;
#endif
}


std::vector<YOLODetection> YOLODetector::infer(const cv::Mat& image) {
#ifdef IS_DEBUG
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double, std::milli> preprocess_time;
    start = std::chrono::high_resolution_clock::now();
#endif
    // 前处理
    PreprocessParams params = this->preprocess(image);
    this->net.setInput(params.blob);
#ifdef IS_DEBUG
    end = std::chrono::high_resolution_clock::now();
    preprocess_time = end - start;
    std::cout << "Preprocessing time: " << preprocess_time.count() << " ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
#endif
    // 正向传播
    std::vector<cv::Mat> outputs;
    this->net.forward(outputs, this->net.getUnconnectedOutLayersNames());
#ifdef IS_DEBUG
    end = std::chrono::high_resolution_clock::now();
    preprocess_time = end - start;
    std::cout << "model forward time: " << preprocess_time.count() << " ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
#endif
    // 后处理
    auto result = this->postProcess(outputs, params);
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

    float scale = std::min(static_cast<float>(this->modelParams.image_size)  / image_w, 
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
    return PreprocessParams(image, blob, top, bottom, left, right);
}


std::vector<YOLODetection> YOLODetector::postProcess(const std::vector<cv::Mat>& outputs, const PreprocessParams& params) {
    std::vector<YOLODetection> results;
    // 检测输出来判断输出模型的模型
    if (outputs.size() == 1) {
        // cls、det、obb、pose 模型
        if (outputs[0].dims == 2) {
            // cls 模型 [1, num_classes]
            std::cout << "Post-processing classification results..." << std::endl;
            results = this->postprocessClassification(outputs);
        }
        else if (outputs[0].dims == 3) {
            // det、obb、pose 模型
            if (this->modelParams.num_classes + 4 == outputs[0].size[1]) {
                // det 模型
                std::cout << "Post-processing detection results..." << std::endl;
                results = this->postprocessDetection(outputs, params);
            }
            else if (this->modelParams.num_classes + 5 == outputs[0].size[1]) {
                // obb 模型
                std::cout << "Post-processing oriented bounding box results..." << std::endl;
                results = this->postprocessOBB(outputs, params);
            }
            else if (this->modelParams.num_classes + 4 + this->modelParams.numKPS * 3 == outputs[0].size[1]) {
                // pose 模型
                std::cout << "Post-processing pose estimation results..." << std::endl;
                results = this->postprocessPose(outputs, params);
            }
            else {
                std::cerr << "Unexpected output dimensions for 3D output: " << outputs[0].size << std::endl;
            }
        }
        else {
            std::cerr << "Unexpected output dimensions: " << outputs[0].dims << std::endl;
        }
    }
    else if (outputs.size() == 2){
        // seg 模型
        std::cout << "Post-processing segmentation results..." << std::endl;
        // results = postProcessSegmentation(outputs, params);
    }
    else {
        std::cerr << "Unexpected number of outputs: " << outputs.size() << std::endl;
    }
    return results;
}


std::vector<YOLODetection> YOLODetector::postprocessClassification(const std::vector<cv::Mat>& outputs) {
    auto output = outputs[0];
    std::vector<std::pair<int, float>> predictions;
    cv::Mat probs;
    cv::exp(output, probs);
    cv::Scalar sum_exp = cv::sum(probs);
    probs /= sum_exp[0]; // 归一化为概率分布
    // 按照概率排序
    for (int i = 0; i < probs.cols; ++i) {
        predictions.emplace_back(i, probs.at<float>(i));
    }
    std::sort(predictions.begin(), predictions.end(), 
                [](const std::pair<int, float>& a, const std::pair<int, float>& b) {return a.second > b.second;});
    // 输出top-5分类结果
    std::vector<YOLODetection> detections;
    for (int i = 0; i < std::min(5, static_cast<int>(predictions.size())); ++i) {
        YOLODetection det;
        det.class_id = predictions[i].first;
        det.confidence = predictions[i].second;
        detections.push_back(det);
    }
    return detections;
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
        float x_scale = static_cast<float>(params.original.cols) / (this->width - params.left - params.right);
        float y_scale = static_cast<float>(params.original.rows) / (this->height - params.top - params.bottom);

        det.box.x = static_cast<int>((det.box.x - params.left) * x_scale);
        det.box.y = static_cast<int>((det.box.y - params.top) * y_scale);
        det.box.width = static_cast<int>(det.box.width * x_scale);
        det.box.height = static_cast<int>(det.box.height * y_scale);
    }
    // 绘制 detections 在图像上
    for (const auto& det : detections) {
        cv::rectangle(params.original, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "ID: " + std::to_string(det.class_id) + " Conf: " + cv::format("%.2f", det.confidence);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(det.box.y, labelSize.height);
        cv::rectangle(params.original, cv::Point(det.box.x, top - labelSize.height),
                      cv::Point(det.box.x + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
        cv::putText(params.original, label, cv::Point(det.box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1);
    }
    cv::imwrite("detection_result.jpg", params.original);
    return detections;
}


std::vector<YOLODetection> YOLODetector::postprocessPose(const std::vector<cv::Mat>& outputs, const PreprocessParams& params) {
    std::vector<YOLODetection> detections;
    auto output = outputs[0];

    const int num_boxes = output.size[2];        // 检测框总数
    // 重塑并转置输出矩阵为 [num_boxes, num_classes + 4 + numKPS * 3] 格式
    cv::Mat output_mat = output.reshape(1, output.size[1]);
    output_mat = output_mat.t();

    // 预分配vector以避免频繁的内存重新分配
    // 根据最大可能的检测数预留空间 (所有框都可能通过阈值)
    std::vector<cv::Rect> all_boxes;
    std::vector<float> all_boxes_confidences;
    std::vector<int> all_boxes_class_ids;
    std::vector<std::vector<PoseKeyPoint>> all_boxes_keypoints;
    all_boxes.reserve(num_boxes);
    all_boxes_confidences.reserve(num_boxes);
    all_boxes_class_ids.reserve(num_boxes);
    all_boxes_keypoints.reserve(num_boxes);

    // 处理每个检测框
    for (int i = 0; i < num_boxes; i++) {
        cv::Mat row = output_mat.row(i);
        float* data = row.ptr<float>();

        // 提取边界框坐标 (中心格式: xc, yc, w, h)
        float xc = data[0];  // 中心x坐标
        float yc = data[1];  // 中心y坐标
        float w = data[2];   // 宽度
        float h = data[3];   // 高度
        int offset_scores = 4;
        int offset_keypoints = 4 + this->modelParams.num_classes;

        // 提取类别置信度分数 (此检测的所有类别)
        cv::Mat scores = row.colRange(offset_scores, offset_keypoints);

        // 提取关键点信息
        std::vector<PoseKeyPoint> keypoints;
        for (int k = 0; k < this->modelParams.numKPS; ++k) {
            float kp_x = data[offset_keypoints + k * 3];
            float kp_y = data[offset_keypoints + k * 3 + 1];
            float kp_conf = data[offset_keypoints + k * 3 + 2];
            keypoints.push_back(PoseKeyPoint{kp_x, kp_y, kp_conf});
        }

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
            all_boxes.emplace_back(cv::Rect(static_cast<int>(x),
                                        static_cast<int>(y),
                                        static_cast<int>(w),
                                        static_cast<int>(h)));
            all_boxes_confidences.emplace_back(static_cast<float>(max_class_score));
            all_boxes_class_ids.emplace_back(class_id_point.x);  // class_id_point.x 包含类别索引
            all_boxes_keypoints.emplace_back(keypoints);
        }
    }

    // 应用非极大值抑制(NMS)消除冗余的重叠框
    // 这确保我们为每个对象只保留最佳检测
    std::vector<int> indices;
    if (!all_boxes.empty()) {
        cv::dnn::NMSBoxes(all_boxes, all_boxes_confidences, 0.0f, this->modelParams.nms_threshold, indices);
    }

    // 将过滤后的索引转换为最终的YOLODetection对象
    detections.reserve(indices.size());  // 预分配最终结果vector
    for (int idx : indices) {
        YOLODetection det;
        det.box = all_boxes[idx];
        det.confidence = all_boxes_confidences[idx];
        det.class_id = all_boxes_class_ids[idx];
        det.keyPoints = all_boxes_keypoints[idx];
        detections.push_back(det);
    }
    for (auto& det : detections) {
        // 调整边界框以补偿前处理中的缩放和填充
        float x_scale = static_cast<float>(params.original.cols) / (this->width - params.left - params.right);
        float y_scale = static_cast<float>(params.original.rows) / (this->height - params.top - params.bottom);

        det.box.x = static_cast<int>((det.box.x - params.left) * x_scale);
        det.box.y = static_cast<int>((det.box.y - params.top) * y_scale);
        det.box.width = static_cast<int>(det.box.width * x_scale);
        det.box.height = static_cast<int>(det.box.height * y_scale);
        for (auto& kp : det.keyPoints) {
            kp.x = (kp.x - params.left) * x_scale;
            kp.y = (kp.y - params.top) * y_scale;
        }
    }
    // 绘制 detections 在图像上
    for (const auto& det : detections) {
        cv::rectangle(params.original, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "ID: " + std::to_string(det.class_id) + " Conf: " + cv::format("%.2f", det.confidence);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(det.box.y, labelSize.height);
        for (const auto& kp : det.keyPoints) {
            if (kp.confidence > 0.5) { // 仅绘制置信度高的关键点 或者 使用 x,y 不为负的条件
                cv::circle(params.original, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)), 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        cv::rectangle(params.original, cv::Point(det.box.x, top - labelSize.height),
                      cv::Point(det.box.x + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
        cv::putText(params.original, label, cv::Point(det.box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1);
    }
    cv::imwrite("detection_result.jpg", params.original);
    return detections;
}


std::vector<YOLODetection> YOLODetector::postprocessSegmentation(const std::vector<cv::Mat>& outputs, const PreprocessParams& params) {
    cv::Mat preds, proto;
    std::cout << params.original.size() << std::endl;
    if (outputs[0].dims == 3 && outputs[1].dims == 4) {
        preds = outputs[0];
        proto = outputs[1];
    } else if (outputs[0].dims == 4 && outputs[1].dims == 3) {
        proto = outputs[0];
        preds = outputs[1];
    } else {
        throw std::runtime_error("Output dimensions mismatch.");
    }

    // Detection: [1, 4+nc+32, 8400] -> [8400, 4+nc+32]
    cv::Mat flat_preds = preds.reshape(1, preds.size[1]); 
    flat_preds = flat_preds.t();

    const int mask_coefficients = 32;
    const int num_boxes = flat_preds.rows;
    const int num_classes = flat_preds.cols - 4 - mask_coefficients; 

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> masks_coeffs;

    for (int i = 0; i < num_boxes; ++i) {
        const float* row_ptr = flat_preds.ptr<float>(i);
        const float* class_scores_ptr = row_ptr + 4;
        float max_class_score = 0.0f;
        int class_id = -1;
        for (int j = 0; j < num_classes; ++j) {
            if (class_scores_ptr[j] > max_class_score) {
                max_class_score = class_scores_ptr[j];
                class_id = j;
            }
        }

        if (max_class_score > this->modelParams.conf_threshold) {
            // 提取边界框坐标 (中心格式: xc, yc, w, h)
            float xc = row_ptr[0];  // 中心 x 坐标
            float yc = row_ptr[1];  // 中心 y 坐标
            float w = row_ptr[2];   // 宽度
            float h = row_ptr[3];   // 高度
            
            // 从中心坐标转换为左上角坐标
            float x = xc - w / 2.0f;  // 左边界
            float y = yc - h / 2.0f;  // 上边界
            
            boxes.emplace_back(cv::Rect(static_cast<int>(x),
                                        static_cast<int>(y),
                                        static_cast<int>(w),
                                        static_cast<int>(h)));
            confidences.push_back(max_class_score);
            class_ids.push_back(class_id);
            
            // 保存 掩膜系数 (Mask Coefficients)
            std::vector<float> coeff(mask_coefficients);
            const float* mask_coeffs_ptr = row_ptr + 4 + num_classes;
            for(int k = 0; k < mask_coefficients; ++k) {
                coeff[k] = mask_coeffs_ptr[k];
            }
            masks_coeffs.push_back(coeff);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.0f, this->modelParams.nms_threshold, indices);

    std::vector<YOLODetection> detections;
    if (indices.empty()) return detections;

    int num_detections = static_cast<int>(indices.size());
    cv::Mat mask_coeffs_mat(num_detections, mask_coefficients, CV_32F);

    // === 优化点 2: 使用 ptr 提升矩阵填充效率，避免 .at() 的边界检查开销 ===
    for (int i = 0; i < num_detections; ++i) {
        int original_idx = indices[i];
        float* mat_row_ptr = mask_coeffs_mat.ptr<float>(i);
        const std::vector<float>& source_coeffs = masks_coeffs[original_idx];
        // 使用 std::copy 更简洁且高效
        std::copy(source_coeffs.begin(), source_coeffs.end(), mat_row_ptr);
    }

    // int proto_h = proto.size[2];
    // int proto_w = proto.size[3];
    // int proto_area = proto_h * proto_w;
    
    // std::vector<int> new_shape = {32, proto_area};
    // cv::Mat proto_flat = proto.reshape(1, new_shape); 

    // cv::Mat raw_masks = mask_coeffs_mat * proto_flat;
    // this->sigmoid(raw_masks);

    // // === 优化点 3: 复用变量，并使用更安全的方式获取mask数据 ===
    // std::vector<int> mask_shape = {proto_h, proto_w};
    // for (int i = 0; i < num_detections; ++i) {
    //     int original_idx = indices[i];
    //     YOLODetection det;
    //     det.box = boxes[original_idx];
    //     det.confidence = confidences[original_idx];
    //     det.class_id = class_ids[original_idx];

    //     // 从 raw_masks 获取第 i 个检测的 mask 数据
    //     // reshape 不会复制数据，只是改变 Mat 的头部信息，非常高效
    //     cv::Mat mask_proto = raw_masks.row(i).reshape(1, mask_shape);
        
    //     cv::Mat mask_up;
    //     cv::resize(mask_proto, mask_up, cv::Size(this->width, this->height), 0, 0, cv::INTER_LINEAR);

    //     // 使用 det.box 进行裁剪，避免创建新变量
    //     det.box.x = std::max(0, det.box.x);
    //     det.box.y = std::max(0, det.box.y);
    //     det.box.width = std::min(det.box.width, this->width - det.box.x);
    //     det.box.height = std::min(det.box.height, this->height - det.box.y);
        
    //     if(det.box.width <= 0 || det.box.height <= 0) {
    //         // 添加一个空的polygon和mask来保持结构一致性
    //         det.polygon.clear();
    //         detections.push_back(det);
    //         continue;
    //     }

    //     cv::Mat mask_roi = mask_up(det.box);
        
    //     cv::Mat mask_binary;
    //     cv::threshold(mask_roi, mask_binary, 0.5, 255, cv::THRESH_BINARY); // 直接二值化为0-255，后续转换为CV_8U
    //     mask_binary.convertTo(mask_binary, CV_8U); 
        
    //     det.polygon = this->mask2Polygon(mask_binary, det.box);
        
    //     // 如果需要，可以在这里填充 mask 成员，但原代码没有使用
    //     // det.mask = mask_binary; 

    //     detections.push_back(det);
    // }
    return detections;
}


std::vector<YOLODetection> YOLODetector::postprocessOBB(const std::vector<cv::Mat>& outputs, const PreprocessParams& params) {
    std::vector<YOLODetection> detections;
    auto output = outputs[0];

    const int num_boxes = output.size[2];        // 检测框总数
    // 重塑并转置输出矩阵为 [num_boxes, xc, yc, w, h + num_classes + rad] 格式
    cv::Mat output_mat = output.reshape(1, output.size[1]);
    output_mat = output_mat.t();

    std::vector<cv::RotatedRect> rotatedBoxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    rotatedBoxes.reserve(num_boxes);
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
        float rad = data[4 + this->modelParams.num_classes]; // 弧度
        float angle = rad * 180.0f / static_cast<float>(CV_PI); // 转换为角度

        // 提取类别置信度分数 (此检测的所有类别)
        cv::Mat scores = row.colRange(4, 4 + this->modelParams.num_classes);

        // 找到置信度最高的类别
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id_point);

        // 过滤掉低于置信度阈值的检测
        if (max_class_score > this->modelParams.conf_threshold) {
            // 将检测结果存储在临时vector中以供NMS处理
            rotatedBoxes.emplace_back(cv::RotatedRect(cv::Point2f(xc, yc), cv::Size2f(w, h), angle));
            confidences.emplace_back(static_cast<float>(max_class_score));
            class_ids.emplace_back(class_id_point.x);  // class_id_point.x 包含类别索引
        }
    }

    std::vector<int> indices;
    if (!rotatedBoxes.empty()) {
        cv::dnn::NMSBoxes(rotatedBoxes, confidences, 0.0f, this->modelParams.nms_threshold, indices);
    }

    // 将过滤后的索引转换为最终的YOLODetection对象
    detections.reserve(indices.size());  // 预分配最终结果vector
    for (int idx : indices) {
        YOLODetection det;
        det.rotatedBox = rotatedBoxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }
    // 调整检测框坐标以补偿前处理的缩放和填充
    for (auto& det : detections) {
        float x_scale = static_cast<float>(params.original.cols) / (this->width - params.left - params.right);
        float y_scale = static_cast<float>(params.original.rows) / (this->height - params.top - params.bottom);

        det.rotatedBox.center.x = (det.rotatedBox.center.x - params.left) * x_scale;
        det.rotatedBox.center.y = (det.rotatedBox.center.y - params.top) * y_scale;
        det.rotatedBox.size.width  = det.rotatedBox.size.width  * x_scale;
        det.rotatedBox.size.height = det.rotatedBox.size.height * y_scale;
    }
    for (const auto& det : detections) {
        // 1. 从 RotatedRect 获取四个顶点
        cv::Point2f pts2f[4];
        det.rotatedBox.points(pts2f);
        std::vector<cv::Point> poly(4);
        for (int i = 0; i < 4; ++i) {
            poly[i] = pts2f[i];
        }

        // 2. 使用 polylines 在原图上绘制旋转框
        std::vector<std::vector<cv::Point>> polys{ poly };
        cv::polylines(params.original, polys, true, cv::Scalar(0, 255, 0), 2);

        // 3. 绘制类别与置信度标签（可选）
        std::string label = "ID: " + std::to_string(det.class_id) +
                            " Conf: " + cv::format("%.2f", det.confidence);
        int min_x = poly[0].x, min_y = poly[0].y;
        for (int i = 1; i < 4; ++i) {
            min_x = std::min(min_x, static_cast<int>(poly[i].x));
            min_y = std::min(min_y, static_cast<int>(poly[i].y));
        }
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(static_cast<int>(min_y), labelSize.height);
        cv::rectangle(params.original,
                      cv::Point(static_cast<int>(min_x), top - labelSize.height),
                      cv::Point(static_cast<int>(min_x) + labelSize.width, top + baseLine),
                      cv::Scalar::all(255), cv::FILLED);
        cv::putText(params.original, label,
                    cv::Point(static_cast<int>(min_x), top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1);
    }
    cv::imwrite("detection_result.jpg", params.original);
    return detections;
}


std::vector<std::string> classification_classes {"tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peacock", "quail", "partridge", "grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern", "crane (bird)", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniels", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland", "Pyrenean Mountain Dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "Griffon Bruxellois", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog", "grey wolf", "Alaskan tundra wolf", "red wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "stick insect", "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral", "ringlet", "monarch butterfly", "small white", "sulfur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram", "bighorn sheep", "Alpine ibex", "hartebeest", "impala", "gazelle", "dromedary", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek", "eel", "coho salmon", "rock beauty", "clownfish", "sturgeon", "garfish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "waste container", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military cap", "beer bottle", "beer glass", "bell-cot", "bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "bow", "bow tie", "brass", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "chest", "chiffonier", "chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "coil", "combination lock", "computer keyboard", "confectionery store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "crane (machine)", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire engine", "fire screen sheet", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "grille", "grocery store", "guillotine", "barrette", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "jack-o'-lantern", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "pulled rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "paper knife", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "speaker", "loupe", "sawmill", "magnetic compass", "mail bag", "mailbox", "tights", "tank suit", "manhole cover", "maraca", "marimba", "mask", "match", "maypole", "maze", "measuring cup", "medicine chest", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "Model T", "modem", "monastery", "monitor", "moped", "mortar", "square academic cap", "mosque", "mosquito net", "scooter", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "nail", "neck brace", "necklace", "nipple", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "packet", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "passenger car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "pitcher", "hand plane", "planetarium", "plastic bag", "plate rack", "plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "billiard table", "soda bottle", "pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler", "running shoe", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT screen", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swimsuit", "swing", "switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vault", "velvet", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "wig", "window screen", "window shade", "Windsor tie", "wine bottle", "wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "yawl", "yurt", "website", "comic book", "crossword", "traffic sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "ice pop", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potato", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard apple", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "cup", "eggnog", "alp", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "shoal", "seashore", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star", "hen-of-the-woods", "bolete", "ear", "toilet paper"};

std::vector<std::string> detection_classes {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

std::vector<std::string> pose_classes {"person"};
std::vector<std::string> obb_classes {"plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"};