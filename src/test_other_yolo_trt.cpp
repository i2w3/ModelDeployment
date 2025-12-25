#include <opencv2/opencv.hpp>
#include "other_yolo.hpp"


int main() {
    trtyolo::InferOption option;
    option.enableSwapRB();  // BGR->RGB转换
    auto detector = std::make_unique<trtyolo::SegmentModel>(
        "/root/ModelDeployment/res/yolo/yolo11s-seg.engine",    // 模型路径
        option                                                  // 推理设置
    );
    const int batch_size = detector->batch();
    std::cout << "batch_size is " << batch_size << std::endl;

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