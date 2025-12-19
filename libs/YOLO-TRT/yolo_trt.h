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
    int max_det;
    // For seg models
    float seg_threshold;
    // For pose models
    int num_kps;
    float kps_threshold;
};


class YOLODetector {
public:
    YOLODetector(const std::string& modelPath, const ModelParams& modelParams);
    std::vector<YOLODetection> infer(const cv::Mat& image);

    PreprocessParams preprocess(const cv::Mat& image);

    // 纯虚函数，由派生类实现不同的后处理逻辑
    virtual std::vector<YOLODetection> postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) = 0;

protected:
    TRTInfer net;
    ModelParams modelParams;
};

// 派生出五个不同的检测器类，以适应不同的模型输出格式
class YOLOCls: public YOLODetector {
public:
    using YOLODetector::YOLODetector; // 没有无参构造函数，必须实现构造函数，使用继承基类构造函数
    std::vector<YOLODetection> postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) override;
};

class YOLODet: public YOLODetector {
public:
    using YOLODetector::YOLODetector;
    std::vector<YOLODetection> postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) override;
};

class YOLOObb: public YOLODetector {
public:
    using YOLODetector::YOLODetector;
    std::vector<YOLODetection> postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) override;
};

class YOLOPose: public YOLODetector {
public:
    using YOLODetector::YOLODetector;
    std::vector<YOLODetection> postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) override;
};

class YOLOSeg: public YOLODetector {
public:
    using YOLODetector::YOLODetector;
    std::vector<YOLODetection> postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) override;
};

#endif // YOLOTRT_H