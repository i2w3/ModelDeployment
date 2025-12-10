#ifndef PADDLE_OCR_UTILS_H
#define PADDLE_OCR_UTILS_H

enum class AngleType {
    ANGLE_0 = 0,
    ANGLE_180 = 1,
};

struct TextDetOutput {
    cv::Mat clip;
    float score;
};

struct TextClsOutput {
    AngleType angle;
    float score;
};

cv::Mat getRotateCropImage(const cv::Mat &src, const std::vector<cv::Point2f> &points);

#endif // PADDLE_OCR_UTILS_H