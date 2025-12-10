#ifndef YOLO_H
#define YOLO_H

#include <chrono>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

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

    PreprocessParams(cv::Mat orig, cv::Mat b, int t, int bo, int l, int r)
        : original(orig), blob(b), top(t), bottom(bo), left(l), right(r) {}
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
    std::vector<YOLODetection> postProcess(const std::vector<cv::Mat>& outputs, const PreprocessParams& params);

private:
    cv::dnn::Net net;
    int width;
    int height;
    ModelParams modelParams;

    std::vector<YOLODetection> postprocessClassification(const std::vector<cv::Mat>& outputs);
    std::vector<YOLODetection> postprocessDetection(const std::vector<cv::Mat>& outputs, const PreprocessParams& params);
    std::vector<YOLODetection> postprocessOBB(const std::vector<cv::Mat>& outputs, const PreprocessParams& params);
    std::vector<YOLODetection> postprocessPose(const std::vector<cv::Mat>& outputs, const PreprocessParams& params);
    std::vector<YOLODetection> postprocessSegmentation(const std::vector<cv::Mat>& outputs, const PreprocessParams& params);
};

#endif // YOLO_H