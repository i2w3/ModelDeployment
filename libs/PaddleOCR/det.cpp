#include "paddleocr.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <limits>

TextDetector::TextDetector(const std::string& modelPath) {
#ifdef IS_DEBUG
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double, std::milli> preprocess_time;
    start = std::chrono::high_resolution_clock::now();
#endif

    std::cout << "Initialized TextDetector with model path: " << modelPath << std::endl;

    this->net = cv::dnn::readNetFromONNX(modelPath);
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);


#ifdef IS_DEBUG
    end = std::chrono::high_resolution_clock::now();
    preprocess_time = end - start;
    std::cout << "Model loading time: " << preprocess_time.count() << " ms" << std::endl;
#endif
}


cv::Mat TextDetector::preprocess(const cv::Mat& input_image) {
    // 将图像大小调整为网络所需的 32 的倍数
    int max_wh = std::max(input_image.rows, input_image.cols);
    int limit_side_len_ = 0;
    if (this->use_minlimit) {
        limit_side_len_ = this->limit_side_len;
    }
    else if (max_wh < 960) {
        limit_side_len_ = 960;
    }
    else if (max_wh < 1500) {

        limit_side_len_ = 1500;
    }
    else {
        limit_side_len_ = 2000;
    }
    float ratio = 0.0f;
    if (!this->use_minlimit) {
        // max limit 模式
        if(max_wh > limit_side_len_) {
            if (input_image.rows > limit_side_len_) {
                ratio = static_cast<float>(limit_side_len_) / input_image.rows;
            }
            else {
                ratio = static_cast<float>(limit_side_len_) / input_image.cols;
            }
        }
        else {
            ratio = 1.0f;
        }
    }
    else {
        // min limit 模式
        if(max_wh > limit_side_len_) {
            if (input_image.rows < limit_side_len_) {
                ratio = static_cast<float>(limit_side_len_) / input_image.rows;
            }
            else {
                ratio = static_cast<float>(limit_side_len_) / input_image.cols;
            }
        }
        else {
            ratio = 1.0f;
        }
    }
    int resize_h = static_cast<int>(input_image.rows * ratio);
    int resize_w = static_cast<int>(input_image.cols * ratio);

    resize_h = static_cast<int>(std::round((resize_h / 32) * 32));
    resize_w = static_cast<int>(std::round((resize_w / 32) * 32));
    cv::Size new_size(resize_w, resize_h);

    cv::Mat resized_image;
    cv::resize(input_image, resized_image, new_size);
#ifdef IS_DEBUG
    // cv::imwrite("resized_det_image.jpg", resized_image);
#endif

    cv::Mat blob;
    cv::dnn::blobFromImage(
        resized_image,
        blob,
        1.0 / 127.5,
        new_size,
        cv::Scalar(127.5, 127.5, 127.5),
        true,
        false
    );
    return blob;
}

std::vector<TextDetOutput> TextDetector::infer(const cv::Mat& input_image) {
#ifdef IS_DEBUG
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double, std::milli> preprocess_time;
    start = std::chrono::high_resolution_clock::now();
#endif
    // 前处理
    cv::Mat blob = this->preprocess(input_image);
    this->net.setInput(blob);
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
    auto result = this->postProcess(outputs, input_image);
#ifdef IS_DEBUG
    end = std::chrono::high_resolution_clock::now();
    preprocess_time = end - start;
    std::cout << "Postprocessing time: " << preprocess_time.count() << " ms" << std::endl;
#endif
    return result;
}

std::vector<TextDetOutput> TextDetector::postProcess(const std::vector<cv::Mat> &outputs, const cv::Mat& original_image) {
    std::vector<TextDetOutput> results;
    cv::Mat image = original_image.clone();
    auto sorcemap = outputs[0].reshape(1, { outputs[0].size[2], outputs[0].size[3] }); // H, W

    // Mark first todo as in_progress
    // 1. Apply threshold to create binary mask
    cv::Mat binary_mask;
    cv::threshold(sorcemap, binary_mask, this->seg_thresh, 255, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8UC1);
    

    // Mark first todo as completed, second as in_progress
    // 2. Apply dilation to connect text regions
    if (this->use_dilation) {
        cv::dilate(binary_mask, binary_mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    }
#ifdef IS_DEBUG
    // cv::imwrite("binary_mask.jpg", binary_mask);
#endif

    // 3. Find contours and convert to bounding boxes
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    int num_contours = std::min(this->max_candidates, static_cast<int>(contours.size()));
    
    std::vector<std::vector<cv::Point2f>> boxes;
    for(size_t i = 0; i < num_contours; i++) {
        const auto& contour = contours[i];
        if (contour.size() < 4) continue;

        // Calculate bounding rectangle
        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
        cv::Point2f vertices[4];
        rotated_rect.points(vertices);

        std::vector<cv::Point2f> box;
        for (int j = 0; j < 4; j++) {
            box.push_back(vertices[j]);
        }
        boxes.push_back(box);
    }
    // plot box on the image
    for (const auto& box : boxes) {
        for (int i = 0; i < 4; i++) {
            cv::line(image, box[i], box[(i+1) % 4], cv::Scalar(0, 255, 0), 2); // 绿色线条，线宽2
        }
    }
#ifdef IS_DEBUG
    cv::imwrite("boxes_on_image.jpg", image);
#endif

    return results;
}

float TextDetector::get_box_score(const cv::Mat &bitmap, const std::vector<cv::Point2f> &box) {
    // 边界检查
    if (bitmap.empty() || box.size() != 4) return 0.0f;

    const int h = bitmap.rows;
    const int w = bitmap.cols;

    // 使用OpenCV的boundingRect进行边界计算 - 更高效
    cv::Point2f boxCopy[4];
    for(int i = 0; i < 4; i++) {
        boxCopy[i] = box[i];
    }

    // 使用OpenCV函数计算边界
    cv::Mat pointsMat(4, 1, CV_32FC2, boxCopy);
    cv::Rect bounding = cv::boundingRect(pointsMat);

    // 裁剪到图像边界
    bounding &= cv::Rect(0, 0, w, h); // 更高效的边界裁剪

    if (bounding.area() <= 0) return 0.0f;

    // 调整box坐标到相对于ROI的坐标
    for(int i = 0; i < 4; i++) {
        boxCopy[i].x -= bounding.x;
        boxCopy[i].y -= bounding.y;
    }

    // 直接在ROI上计算 - 避免复制
    cv::Mat roi = bitmap(bounding);

    // 使用预分配的mask (缓存机制)
    thread_local cv::Mat cachedMask;
    if (cachedMask.size() != bounding.size() || cachedMask.type() != CV_8UC1) {
        cachedMask = cv::Mat::zeros(bounding.size(), CV_8UC1);
    } else {
        cachedMask.setTo(0);
    }

    // 安全的多边形填充 - 确保坐标在mask范围内
    cv::Point contour[4];
    for(int i = 0; i < 4; i++) {
        // 将坐标限制在mask边界内
        int x = std::max(0, std::min(bounding.width - 1, static_cast<int>(boxCopy[i].x)));
        int y = std::max(0, std::min(bounding.height - 1, static_cast<int>(boxCopy[i].y)));
        contour[i] = cv::Point(x, y);
    }
    cv::fillConvexPoly(cachedMask, contour, 4, cv::Scalar(255), 8, 0);

    // 使用SIMD优化的计算
    cv::Scalar meanVal;
    cv::meanStdDev(roi, meanVal, cv::noArray(), cachedMask);

    return static_cast<float>(meanVal[0]) / 255.0f; // 归一化
}
