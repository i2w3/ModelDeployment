#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"


void visualize(cv::Mat& image, trtyolo::SegmentRes& result, const std::vector<std::string>& labels) {
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
        cv::Mat float_mask(result.masks[i].height, result.masks[i].width, CV_32FC1, result.masks[i].data.data());
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


int main() {
    trtyolo::InferOption option;
    option.enableSwapRB();  // BGR->RGB转换
    auto detector = std::make_unique<trtyolo::SegmentModel>(
        "/root/ModelDeployment/res/yolo/yolo11s-seg.engine",  // 模型路径
        option                         // 推理设置
    );

    cv::Mat cv_image = cv::imread("/root/ModelDeployment/res/bus.jpg");
    if (cv_image.empty()) {
        throw std::runtime_error("无法加载测试图片");
    }
    trtyolo::Image input_image(
        cv_image.data,  // 像素数据指针
        cv_image.cols,  // 图像宽度
        cv_image.rows   // 图像高度
    );
    trtyolo::SegmentRes result = detector->predict(input_image);
    std::cout << result << std::endl;
    visualize(cv_image, result, {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                                "cell phone",  "microwave",  "oven",  "toaster",  "sink",  "refrigerator",  "book",
                                "clock",  "vase",  "scissors",  "teddy bear",  "hair drier",  "toothbrush"});
    cv::imwrite("result_seg.jpg", cv_image);
}