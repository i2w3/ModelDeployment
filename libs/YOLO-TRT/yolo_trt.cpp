#include "yolo_trt.h"

YOLODetector::YOLODetector(const std::string& modelPath, const ModelParams& modelParams_) 
    :net(modelPath), modelParams(modelParams_) {
    assert(modelParams.num_classes == modelParams.class_names.size());
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
    // letter box resize
    int image_h = image.rows;
    int image_w = image.cols;

    float scale = std::min(static_cast<float>(this->modelParams.image_size) / image_w, 
                           static_cast<float>(this->modelParams.image_size) / image_h);
    int resized_w = static_cast<int>(image_w * scale);
    int resized_h = static_cast<int>(image_h * scale);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h));

    int pad_w = this->modelParams.image_size - resized_w;
    int pad_h = this->modelParams.image_size - resized_h;

    float dw = pad_w / 2.0f;
    float dh = pad_h / 2.0f;
    int top     = static_cast<int>(std::round(dh - 0.1f));
    int bottom  = static_cast<int>(std::round(dh + 0.1f));
    int left    = static_cast<int>(std::round(dw - 0.1f));
    int right   = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, 
                       top, bottom, left, right, 
                       cv::BORDER_CONSTANT, 
                       cv::Scalar(114, 114, 114));

    cv::Mat blob = cv::dnn::blobFromImage(padded, 
                                          1.0 / 255.0, 
                                          cv::Size(this->modelParams.image_size, this->modelParams.image_size), 
                                          cv::Scalar(), 
                                          true, 
                                          false);
    return PreprocessParams(image, blob, top, bottom, left, right, scale);
}


void plot_result(cv::Mat &image, const std::vector<YOLODetection> &result, const ModelParams &model_params) {
    cv::Point default_pos = cv::Point(10, 10);
    cv::Point label_pos(image.cols, image.rows);
    cv::Mat mask = image.clone();
    for (const auto& det : result) {
        if(!det.box.empty()) {
            // HBB box
            cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
            label_pos = cv::Point(det.box.x, det.box.y);
        }
        else if(!det.rotatedBox.size.empty()) {
            // OBB box
            cv::Point2f vertices[4];
            det.rotatedBox.points(vertices);
            for (int j = 0; j < 4; j++) {
                cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
                if (label_pos.y > vertices[j].y) {
                    // 更新为最靠上的点
                    label_pos.x = static_cast<int>(vertices[j].x);
                    label_pos.y = static_cast<int>(vertices[j].y);
                }
            }
        }
        else {
            // CLS 默认位置
            label_pos = default_pos;
            default_pos.y += 20;
        }
        std::string label = "ID: " + model_params.class_names[det.class_id] + " Conf: " + cv::format("%.4f", det.confidence);
        cv::putText(image, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
        cv::putText(image, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        // 绘制关键点
        bool is_pose = false;
        for (const auto& kp : det.keyPoints) {
            if (kp.x >= 0 && kp.y >= 0) { // 仅绘制有效关键点，后处理时将低于置信度的点置为负数，也可以用 kp.confidence 判断
                cv::circle(image, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)), 3, cv::Scalar(0, 0, 255), -1);
            }
            is_pose = true;
        }
        if (is_pose) {
            // 绘制骨架连接线
            for (const auto& bone : skeleton_pairs) {
                int kp1_idx = bone.first;
                int kp2_idx = bone.second;
                if (det.keyPoints[kp1_idx].confidence > 0.5 && det.keyPoints[kp2_idx].confidence > 0.5) {
                    cv::Point p1((int)det.keyPoints[kp1_idx].x, (int)det.keyPoints[kp1_idx].y);
                    cv::Point p2((int)det.keyPoints[kp2_idx].x, (int)det.keyPoints[kp2_idx].y);
                    cv::line(image, p1, p2, cv::Scalar(0, 0x27, 0xC1), 2);
                }
            }
        }
        // 绘制 mask
        if (!det.boxMask.empty()) {
            int colorIndex = det.class_id % color_list.size();
            cv::Scalar color = cv::Scalar(color_list[colorIndex][0], color_list[colorIndex][1], color_list[colorIndex][2]);
            mask(det.box).setTo(color * 255, det.boxMask);
        }
    }
    cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
}