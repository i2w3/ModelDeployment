#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cstdio>

#include "yolo_trt.h"

std::vector<std::string> detection_classes {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

ModelParams detParams  = {640, 80, detection_classes, 0.25f, 0.45f, 300, 0, 0, 0.0f};

std::string getEnvironmentVariable(const std::string_view& varname) {
    // Source - https://stackoverflow.com/questions/15916695/can-anyone-give-me-example-code-of-dupenv-s
    // Posted by chrisake, modified by community. License - CC BY-SA 4.0
    char* buf = nullptr;
    size_t sz = 0;
    if (_dupenv_s(&buf, &sz, varname.data()) != 0 || buf == nullptr) {
        throw std::runtime_error("Failed to get environment variable: " + std::string(varname));
    }
    std::string result(buf);
    free(buf);
    return result;
}

void setupEnv() {
    // Disable OpenCV logging
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // Set GStreamer environment variables
    std::string vcpkg_root(getEnvironmentVariable("VCPKG_ROOT")), gstreamer_plugins, gstreamer_bin;
    std::filesystem::path gstreamer_plugins_path, gstreamer_bin_path;
    // only test pass on Windows
    #ifdef IS_DEBUG
        std::cout << cv::getBuildInformation() << std::endl;
        _putenv("GST_DEBUG=3");
        gstreamer_plugins_path = std::filesystem::path(vcpkg_root) / "installed" / "x64-windows" / "debug" / "plugins" / "gstreamer";
        gstreamer_bin_path = std::filesystem::path(vcpkg_root) / "installed" / "x64-windows" / "debug" / "bin";
    #else
        gstreamer_plugins_path = std::filesystem::path(vcpkg_root) / "installed" / "x64-windows" / "plugins" / "gstreamer";
        gstreamer_bin_path = std::filesystem::path(vcpkg_root) / "installed" / "x64-windows" / "bin";
    #endif
    gstreamer_plugins = "GST_PLUGIN_PATH=" + gstreamer_plugins_path.string();
    gstreamer_bin = "PATH=" + gstreamer_bin_path.string() + ";" + getEnvironmentVariable("PATH");
    _putenv(gstreamer_plugins.c_str());
    _putenv(gstreamer_bin.c_str());
}


int main() {
    setupEnv();
    std::string rtspInputUrl, rtspOutputUrl, model_path;

    // std::cout << "Enter the path to the model file: ";
    // std::cin >> model_path;
    model_path = "D:/code/ModelDeployment/res/yolo/trt_cache/yolo11n_fp16_16084721098685413204_0_fp16_sm89.engine";

    YOLODet model(model_path, detParams);

    rtspInputUrl = "rtspsrc location=rtsp://127.0.0.1:8554/live protocols=tcp latency=200 ! "
                   "rtph264depay wait-for-keyframe=true ! h264parse ! avdec_h264 ! "
                   "videoconvert ! video/x-raw, format=BGR ! appsink drop=true";
    rtspOutputUrl = "appsrc is-live=true format=time ! videoconvert ! x264enc tune=zerolatency "
                    "speed-preset=ultrafast bitrate=2000 ! rtspclientsink "
                    "location=rtsp://127.0.0.1:8554/out protocols=tcp";

    cv::VideoCapture cap(rtspInputUrl, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cout << "Failed to open RTSP stream!" << std::endl;
        return -1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Stream properties: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;

    cv::VideoWriter writer(rtspOutputUrl, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);

    if (!writer.isOpened()) {
        std::cerr << "Failed to open VideoWriter" << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::chrono::steady_clock::time_point start, end;
    std::string fps_time;
    cv::Point fps_pos = cv::Point(20, 40);
    while (true) {
        start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Empty frame! Stop streaming." << std::endl;
            break;
        }
        auto result = model.infer(frame);
        for (const auto& det : result) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "ID: " + detParams.class_names[det.class_id] + " Conf: " + cv::format("%.2f", det.confidence);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(det.box.y, labelSize.height);
            cv::rectangle(frame, cv::Point(det.box.x, top - labelSize.height),
                        cv::Point(det.box.x + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
            cv::putText(frame, label, cv::Point(det.box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1);
        }
        end = std::chrono::high_resolution_clock::now();
        fps_time = cv::format("FPS: %.2f", 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()),
        cv::putText(frame, fps_time, fps_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 4);
        cv::putText(frame, fps_time, fps_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        writer.write(frame);
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}