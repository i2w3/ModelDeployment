#ifndef YOLOTRT_H
#define YOLOTRT_H

#include <chrono>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "TRTInfer.h"

struct PoseKeyPoint {
	float x = 0;
	float y = 0;
	float confidence = 0;
};

struct YOLODetection {
    int class_id;
    float confidence;
    cv::Rect box;
    cv::RotatedRect rotatedBox;
    cv::Mat boxMask;
    std::vector<PoseKeyPoint> keyPoints;
};

struct PreprocessParams {
    cv::Mat original;
    cv::Mat blob;
    int top;
    int bottom;
    int left;
    int right;
    float scale;

    PreprocessParams(cv::Mat orig, cv::Mat b, int t, int bo, int l, int r, float s)
        : original(orig), blob(b), top(t), bottom(bo), left(l), right(r), scale(s) {}
};


struct ModelParams {
    // Common params
    int image_size;
    int num_classes;
    std::vector<std::string> class_names;
    float conf_threshold;
    float nms_threshold;
    // For seg models
    float seg_threshold;
    // For pose models
    int numKPS;
    float kps_threshold;
};


class YOLODetector {
public:
    YOLODetector(const std::string& modelPath, const ModelParams& modelParams);
    std::vector<YOLODetection> infer(const cv::Mat& image);

    PreprocessParams preprocess(const cv::Mat& image);
    std::vector<YOLODetection> postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params);

private:
    TRTInfer net;
    ModelParams modelParams;

    std::vector<YOLODetection> postprocessDetection(const std::vector<cv::Mat>& outputs, const PreprocessParams& params);
};

#endif // YOLOTRT_H