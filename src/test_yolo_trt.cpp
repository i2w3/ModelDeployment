#include "yolo_trt.h"

// 定义在 yolo_trt.h 中
extern const ModelParams clsParams, detParams, obbParams, poseParams, segParams;

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::string model_type, model_path, image_path;
    std::unique_ptr<YOLODetector> model;
    ModelParams model_params;

    std::cout << "Enter the type of the model file (cls/det/obb/pose/seg): ";
    std::cin >> model_type;
    if (model_type != "cls" && model_type != "det" && model_type != "obb" && model_type != "pose" && model_type != "seg") {
        std::cerr << "Unsupported model type!" << std::endl;
        return -1;
    }

    std::cout << "Enter the path to the model file: ";
    std::cin >> model_path;

    if (model_type == "cls") {
        std::cout << "Loading CLS model..." << std::endl;
        model_params = clsParams;
        model = std::make_unique<YOLOCls>(model_path, clsParams);
    }
    else if (model_type == "det") {
        std::cout << "Loading DET model..." << std::endl;
        model_params = detParams;
        model = std::make_unique<YOLODet>(model_path, detParams);
    }
    else if (model_type == "obb") {
        std::cout << "Loading OBB model..." << std::endl;
        model_params = obbParams;
        model = std::make_unique<YOLOObb>(model_path, obbParams);
    }
    else if (model_type == "pose") {
        std::cout << "Loading POSE model..." << std::endl;
        model_params = poseParams;
        model = std::make_unique<YOLOPose>(model_path, poseParams);
    }
    else if (model_type == "seg") {
        std::cout << "Loading SEG model..." << std::endl;
        model_params = segParams;
        model = std::make_unique<YOLOSeg>(model_path, segParams);
    }

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