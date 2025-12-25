#ifndef OTHER_YOLO_HPP
#define OTHER_YOLO_HPP

#include <variant>

#include "utils.hpp"
#include "trtyolo.hpp"

using ModelVariant = std::variant<
    std::monostate,
    std::unique_ptr<trtyolo::ClassifyModel>,
    std::unique_ptr<trtyolo::DetectModel>,
    std::unique_ptr<trtyolo::OBBModel>,
    std::unique_ptr<trtyolo::PoseModel>,
    std::unique_ptr<trtyolo::SegmentModel>
>;

std::pair<ModelVariant, ModelParams> omodel_parse(const std::string& model_type, const std::string& model_path) {
    // 定义在 yolo_trt.h 中
    extern const ModelParams clsParams, detParams, obbParams, poseParams, segParams;

    trtyolo::InferOption option;
    option.enableSwapRB();  // BGR->RGB转换

    if (model_type == "cls") {
        std::cout << "Loading CLS model..." << std::endl;
        return {std::make_unique<trtyolo::ClassifyModel>(model_path, option), clsParams};
    }
    else if (model_type == "det") {
        std::cout << "Loading DET model..." << std::endl;
        return {std::make_unique<trtyolo::DetectModel>(model_path, option), detParams};
    }
    else if (model_type == "obb") {
        std::cout << "Loading OBB model..." << std::endl;
        return {std::make_unique<trtyolo::OBBModel>(model_path, option), obbParams};
    }
    else if (model_type == "pose") {
        std::cout << "Loading POSE model..." << std::endl;
        return {std::make_unique<trtyolo::PoseModel>(model_path, option), poseParams};
    }
    else if (model_type == "seg") {
        std::cout << "Loading SEG model..." << std::endl;
        return {std::make_unique<trtyolo::SegmentModel>(model_path, option), segParams};
    }
    else {
        std::cerr << "Unsupported model type!" << std::endl;
        return {std::monostate{}, ModelParams{}};
    }
}


void visualize(cv::Mat& image, const trtyolo::ClassifyRes& result, const std::vector<std::string>& labels) {
    for (int i = 0; i < result.num; ++i) {
        int         cls       = result.classes[i];
        float       score     = result.scores[i];
        auto&       label     = labels[cls];
        std::string labelText = label + " " + cv::format("%.3f", score);

        // Draw rectangle and label
        cv::putText(image, labelText, cv::Point(5, 32 + i * 32), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(251, 81, 163), 1);
    }
}


void visualize(cv::Mat& image, const trtyolo::DetectRes& result, const std::vector<std::string>& labels) {
    for (int i = 0; i < result.num; ++i) {
        const auto& box        = result.boxes[i];
        int         cls        = result.classes[i];
        float       score      = result.scores[i];
        const auto& label      = labels[cls];
        std::string label_text = label + " " + cv::format("%.3f", score);

        // 绘制矩形和标签
        int      base_line;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - label_size.height), cv::Point(box.left + label_size.width, box.top), cv::Scalar(125, 40, 81), -1);
        cv::putText(image, label_text, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);
    }
}


void visualize(cv::Mat& image, const trtyolo::OBBRes& result, const std::vector<std::string>& labels) {
    for (int i = 0; i < result.num; ++i) {
        auto       xyxyxyxy   = result.boxes[i].xyxyxyxy();               // 当前边界框
        int         cls        = result.classes[i];                        // 当前类别
        float       score      = result.scores[i];                         // 当前置信度
        auto&       label      = labels[cls];                              // 获取类别标签
        std::string label_text = label + " " + cv::format("%.3f", score);  // 构造显示的标签文本

        // 计算标签的尺寸
        int      base_line;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);

        // 绘制旋转矩形
        std::vector<cv::Point> points = {cv::Point(xyxyxyxy[0], xyxyxyxy[1]), cv::Point(xyxyxyxy[2], xyxyxyxy[3]), cv::Point(xyxyxyxy[4], xyxyxyxy[5]), cv::Point(xyxyxyxy[6], xyxyxyxy[7])};
        cv::polylines(image, points, true, cv::Scalar(251, 81, 163), 2, cv::LINE_AA);

        // 绘制标签背景
        cv::rectangle(image, cv::Point(xyxyxyxy[0], xyxyxyxy[1] - label_size.height),
                      cv::Point(xyxyxyxy[0] + label_size.width, xyxyxyxy[1]),
                      cv::Scalar(125, 40, 81), -1);

        // 绘制标签文本
        cv::putText(image, label_text, cv::Point(xyxyxyxy[0], xyxyxyxy[1] - base_line), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);
    }
}


void visualize(cv::Mat& image, const trtyolo::PoseRes& result, const std::vector<std::string>& labels) {
    // 定义人体关键点连接关系（骨架）
    std::vector<std::pair<int, int>> skeleton = {
        {16, 14},
        {14, 12},
        {17, 15},
        {15, 13},
        {12, 13},
        { 6, 12},
        { 7, 13},
        { 6,  7},
        { 6,  8},
        { 7,  9},
        { 8, 10},
        { 9, 11},
        { 2,  3},
        { 1,  2},
        { 1,  3},
        { 2,  4},
        { 3,  5},
        { 4,  6},
        { 5,  7}
    };

    // 遍历每个检测到的目标
    for (int i = 0; i < result.num; ++i) {
        auto&       box        = result.boxes[i];                          // 当前目标的边界框
        int         cls        = result.classes[i];                        // 当前目标的类别
        float       score      = result.scores[i];                         // 当前目标的置信度
        auto&       label      = labels[cls];                              // 获取类别对应的标签
        std::string label_text = label + " " + cv::format("%.3f", score);  // 构造显示的标签文本

        // 绘制边界框和标签
        int      base_line;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - label_size.height), cv::Point(box.left + label_size.width, box.top), cv::Scalar(125, 40, 81), -1);
        cv::putText(image, label_text, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);

        // 获取当前目标的关键点数量
        int  num_keypoints = result.kpts[i].size();
        bool is_pose       = num_keypoints == 17;  // 判断是否为人体姿态检测结果（17个关键点）

        // 绘制关键点
        for (int j = 0; j < num_keypoints; ++j) {
            auto& kpt = result.kpts[i][j];  // 当前关键点
            if (kpt.conf.has_value() && kpt.conf.value() < 0.25) {
                // 如果关键点的置信度低于阈值，跳过绘制
                continue;
            }
            if (int(kpt.x) % image.cols != 0 && int(kpt.y) % image.rows != 0) {
                // 绘制关键点
                cv::circle(image, cv::Point(kpt.x, kpt.y), 3, cv::Scalar(125, 40, 81), -1, cv::LINE_AA);
            }
        }

        // 绘制关键点连接线（骨架）
        if (is_pose) {
            for (const auto& sk : skeleton) {
                const auto& kpt1 = result.kpts[i][sk.first - 1];   // 第一个关键点
                const auto& kpt2 = result.kpts[i][sk.second - 1];  // 第二个关键点

                // 如果关键点的置信度低于阈值，跳过绘制
                if (kpt1.conf < 0.25 || kpt2.conf < 0.25) {
                    continue;
                }

                // 检查关键点是否超出图像边界
                if (int(kpt1.x) % image.cols == 0 || int(kpt1.y) % image.rows == 0 || int(kpt1.x) < 0 || kpt1.y < 0) {
                    continue;
                }
                if (int(kpt2.x) % image.cols == 0 || int(kpt2.y) % image.rows == 0 || int(kpt2.x) < 0 || kpt2.y < 0) {
                    continue;
                }

                // 绘制连接线
                cv::line(image, cv::Point(kpt1.x, kpt1.y), cv::Point(kpt2.x, kpt2.y), cv::Scalar(253, 168, 208), 2, cv::LINE_AA);
            }
        }
    }
}


void visualize(cv::Mat& image, const trtyolo::SegmentRes& result, const std::vector<std::string>& labels) {
    int im_h = image.rows;  // 图像高度
    int im_w = image.cols;  // 图像宽度

    // 遍历每个检测到的目标
    for (int i = 0; i < result.num; ++i) {
        auto&       box        = result.boxes[i];                          // 当前目标的边界框
        int         cls        = result.classes[i];                        // 当前目标的类别
        float       score      = result.scores[i];                         // 当前目标的置信度
        auto&       label      = labels[cls];                              // 获取类别对应的标签
        std::string label_text = label + " " + cv::format("%.3f", score);  // 构造显示的标签文本

        // 绘制边界框和标签
        int      base_line;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - label_size.height), cv::Point(box.left + label_size.width, box.top), cv::Scalar(125, 40, 81), -1);
        cv::putText(image, label_text, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);

        // 创建分割掩码
        auto xyxy = box.xyxy();
        int  w    = std::max(xyxy[2] - xyxy[0] + 1, 1);
        int  h    = std::max(xyxy[3] - xyxy[1] + 1, 1);

        int x1 = std::max(0, xyxy[0]);
        int y1 = std::max(0, xyxy[1]);
        int x2 = std::min(im_w, xyxy[2] + 1);
        int y2 = std::min(im_h, xyxy[3] + 1);

        // 将模型输出的浮点掩码转换为 OpenCV 的 Mat 格式
        cv::Mat float_mask(result.masks[i].height, result.masks[i].width, CV_32FC1, const_cast<float*>(result.masks[i].data.data()));
        cv::resize(float_mask, float_mask, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

        // 将浮点掩码转换为布尔掩码
        cv::Mat bool_mask;
        cv::threshold(float_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);

        // 创建一个与原图大小相同的掩码图像
        cv::Mat mask_image = cv::Mat::zeros(image.size(), CV_8UC1);

        // 计算从 bool_mask 中裁切的偏移（若 xyxy 左上角为负，则偏移为负值的绝对值）
        int src_x_offset = std::max(0, -xyxy[0]);
        int src_y_offset = std::max(0, -xyxy[1]);

        // 目标区域的宽高（在原图中的实际可用区域）
        int target_w = x2 - x1;
        int target_h = y2 - y1;
        if (target_w <= 0 || target_h <= 0)
            continue;

        // 确保源裁切区域不会超出 bool_mask 的边界
        if (src_x_offset + target_w > bool_mask.cols)
            target_w = bool_mask.cols - src_x_offset;
        if (src_y_offset + target_h > bool_mask.rows)
            target_h = bool_mask.rows - src_y_offset;
        if (target_w <= 0 || target_h <= 0)
            continue;

        // 定义目标区域和源区域
        cv::Rect target_rect(x1, y1, target_w, target_h);                      // 目标区域在 mask_image 中的位置
        cv::Rect source_rect(src_x_offset, src_y_offset, target_w, target_h);  // 源区域在 bool_mask 中的位置

        // 将 bool_mask 的指定区域复制到 mask_image 的指定区域
        bool_mask(source_rect).copyTo(mask_image(target_rect));

        // 创建一个与原图大小相同的颜色图像
        cv::Mat color_image(image.size(), image.type(), cv::Scalar(251, 81, 163));

        // 使用掩码将颜色图像与原图进行混合
        cv::Mat masked_color_image;
        cv::bitwise_and(color_image, color_image, masked_color_image, mask_image);

        cv::addWeighted(image, 1.0, masked_color_image, 0.5, 0, image);
    }
}

// 存储图像数据与预处理后数据的配对
struct OPreprocessedImage {
    cv::Mat image;
    trtyolo::Image data;
};


// 存储图像数据与推理结果的配对
struct OInferenceResult {
    cv::Mat image;
    trtyolo::SegmentRes result;
};

#endif // OTHER_YOLO_HPP