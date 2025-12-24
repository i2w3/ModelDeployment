#include "utils.hpp"

// 测试 yolo trt模型
int main(int argc, char** argv) {
    setupEnv();
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <trtModelType> <trtModelPath>"
                  << std::endl;
        return -1;
    }
    std::string model_type(argv[1]), model_path(argv[2]), image_path;
    auto model_pair = model_parse(model_type, model_path);
    if (!model_pair.first) {
        return -1;
    }
    auto& model = model_pair.first;
    auto& model_params = model_pair.second;

    while (true) {
        std::cout << "Enter the path to the image file: ";
        std::cin >> image_path;
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Could not open or find the image!" << std::endl;
            return -1;
        }
        auto start = std::chrono::high_resolution_clock::now();
        auto result = model->infer(image);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end - start;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
        std::cout << "Detect " << result.size() << " objects." << std::endl;
        plot_result(image, result, model_params);
        cv::imwrite("output.jpg", image);
    }
    return 0;
}