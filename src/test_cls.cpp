#include "paddleocr.h"

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    
    std::string model_path, image_path;
    std::cout << "Enter the path to the model file: ";
    std::cin >> model_path;
    std::cout << "Enter the path to the image file: ";
    std::cin >> image_path;

    TextClassifier model(model_path);
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    AngleType angle = model.infer(image);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end - start;
    std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
    std::cout << "Predicted angle: " << (angle == AngleType::ANGLE_0 ? "0" : "180") << std::endl;
}