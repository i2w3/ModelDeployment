#include "yolo.h"

// 定义在 libs/YOLO/yolo.cpp 的末尾
extern std::vector<std::string> classification_classes;
extern std::vector<std::string> detection_classes;
extern std::vector<std::string> obb_classes;
extern std::vector<std::string> pose_classes;

ModelParams clsParams = {224, 1000, classification_classes, 0.0f, 0.0f, 0, 0, 0.0f};
ModelParams detParams  = {640, 80, detection_classes, 0.25f, 0.45f, 0, 0, 0.0f};
ModelParams obbParams = {1024, 15, obb_classes, 0.25f, 0.45f, 0, 0, 0.25f};
ModelParams poseParams = {640, 1, pose_classes, 0.25f, 0.45f, 0, 17, 0.25f};


int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    
    std::string model_path, image_path;
    std::cout << "Enter the path to the model file: ";
    std::cin >> model_path;
    std::cout << "Enter the path to the image file: ";
    std::cin >> image_path;

    YOLODetector model(model_path, obbParams);
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
    return 0;
}