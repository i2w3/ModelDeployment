#include "yolo_trt.h"

std::vector<YOLODetection> YOLOCls::postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) {
    std::vector<YOLODetection> detections;
    detections.reserve(this->modelParams.max_det);
    auto output = outputs.at("output0").row(0); // [1, num_classes] -> [num_classes]

    auto _ = params; // 未使用参数，避免编译警告

    // Softmax 归一化
    cv::Mat probs;
    cv::exp(output, probs);
    cv::Scalar sum_exp = cv::sum(probs);
    probs /= sum_exp[0];

    // 预分配 vector 以避免频繁的内存重新分配
    std::vector<std::pair<int, float>> predictions;
    predictions.reserve(probs.cols);
    for (int i = 0; i < probs.cols; ++i) {
        predictions.emplace_back(i, probs.at<float>(0, i));
    }
    std::sort(predictions.begin(), predictions.end(), 
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {return a.second > b.second;});

    // 输出 top-K 分类结果
    for (int i = 0; i < std::min(this->modelParams.max_det, static_cast<int>(predictions.size())); ++i) {
        YOLODetection det;
        det.class_id = predictions[i].first;
        det.confidence = predictions[i].second;
        detections.push_back(det);
    }
    return detections;
}