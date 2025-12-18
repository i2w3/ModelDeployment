#include "yolo_trt.h"

std::vector<std::string> detection_classes {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

ModelParams detParams  = {640, 80, detection_classes, 0.25f, 0.45f, 0, 0, 0.0f};


int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    
    std::string model_path, image_path;
    std::cout << "Enter the path to the model file: ";
    std::cin >> model_path;
    YOLODetector model(model_path, detParams);

    while (true) {
        std::cout << "Enter the path to the image file: ";
        std::cin >> image_path;
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Could not open or find the image!" << std::endl;
            return -1;
        }
        auto start = std::chrono::high_resolution_clock::now();
        auto result = model.infer(image);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end - start;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
        std::cout << "Detect " << result.size() << " objects." << std::endl;
        // 绘制 detections 在图像上
        #ifdef IS_DEBUG
        for (const auto& det : result) {
            cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "ID: " + std::to_string(det.class_id) + " Conf: " + cv::format("%.2f", det.confidence);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(det.box.y, labelSize.height);
            cv::rectangle(image, cv::Point(det.box.x, top - labelSize.height),
                        cv::Point(det.box.x + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
            cv::putText(image, label, cv::Point(det.box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1);
        }
        cv::imshow("Detections", image);
        cv::waitKey(0);
        #endif
    }
    return 0;
}