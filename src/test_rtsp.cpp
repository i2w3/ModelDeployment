#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cstdio>

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
    std::string rtspInputUrl, rtspOutputUrl;

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
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Empty frame" << std::endl;
            break;
        }
        writer.write(frame);
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}