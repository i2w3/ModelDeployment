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