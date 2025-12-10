#include <opencv2/opencv.hpp>

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    
    std::string image_path;
    std::cout << "Enter the path to the image file: ";
    std::cin >> image_path;
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }
    cv::imshow("Display window", image);
    cv::waitKey(0);
    return 0;
}