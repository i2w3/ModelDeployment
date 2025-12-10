#ifndef PADDLE_OCR_PADDLEOCR_H
#define PADDLE_OCR_PADDLEOCR_H

#include <chrono>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"

class TextDetector {
public:
    TextDetector(const std::string& modelPath);
    cv::Mat preprocess(const cv::Mat& input_image);
    std::vector<TextDetOutput> infer(const cv::Mat& input_image);
    std::vector<TextDetOutput> postProcess(const std::vector<cv::Mat> &outputs, const cv::Mat& original_image);

private:
    cv::dnn::Net net;
    float seg_thresh = 0.3f;
    float box_thresh = 0.7f;
    int max_candidates = 1000;
    float unclip_ratio = 2.0;
    bool use_fastscore = true;
    bool use_dilation = false;
    int limit_side_len = 736;
    bool use_minlimit = true;
    int min_size = 3;

    float get_box_score(const cv::Mat &bitmap, const std::vector<cv::Point2f> &box);
};





    





// class Model {
// public:
//     Model(const std::string& modelPath, int height_, int width_)
//         : height(height_), width(width_) {
//         this->net = cv::dnn::readNetFromONNX(modelPath);
//         this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//         this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//     }

// protected:
//     cv::dnn::Net net;
//     int height;
//     int width;

//     cv::Mat preprocess(const cv::Mat& input_image) const {
//         // 1. 确保输入图像是 3 通道
//         cv::Mat image;
//         if (input_image.channels() == 1) {
//             cv::cvtColor(input_image, image, cv::COLOR_GRAY2BGR);
//         } else {
//             image = input_image;
//         }

//         // 2. 比例计算
//         // 保持高度固定为 this->height，计算新的宽度
//         float ratio = (float)image.cols / (float)image.rows;
//         int resize_w = (int)std::ceil(this->height * ratio); // 向上取整数
//         // 宽度截断：如果计算出的宽度超过了模型输入宽度，强制截断
//         if (resize_w > this->width) {
//             resize_w = this->width;
//         }

//         // 3. Resize
//         cv::Mat resized;
//         cv::resize(image, resized, cv::Size(resize_w, this->height));

//         // 4. Padding：建立一个全黑的画布 (Height, Width, 3通道)
//         cv::Mat padded = cv::Mat::zeros(this->height, this->width, CV_8UC3);
//         resized.copyTo(padded(cv::Rect(0, 0, resize_w, this->height)));
//     #ifdef IS_DEBUG
//         cv::imwrite("debug_padded.jpg", padded);
//     #endif

//         // 5. 生成 Blob
//         cv::Mat blob;
//         cv::dnn::blobFromImage(
//             padded, 
//             blob, 
//             1.0 / 127.5, 
//             cv::Size(this->width, this->height), 
//             cv::Scalar(127.5, 127.5, 127.5), 
//             true, 
//             false
//         );
//         return blob;
//     }
// };


// class TextClassifier : public Model {
// public:
//     explicit TextClassifier(const std::string& modelPath);
//     AngleType infer(const cv::Mat& input_image);
//     AngleType postProcess(const std::vector<cv::Mat> &outputs);
// };
#endif // PADDLE_OCR_PADDLEOCR_H