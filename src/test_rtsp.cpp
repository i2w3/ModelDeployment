#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cstdio>

// GSTREAMER 取流和推流，目测目前延迟较低
std::string getEnvironmentVariable(const std::string& varname) {
    #ifdef _WIN32
    // Source - https://stackoverflow.com/questions/15916695/can-anyone-give-me-example-code-of-dupenv-s
    // Posted by chrisake, modified by community. License - CC BY-SA 4.0
    char* buf = nullptr;
    size_t sz = 0;
    if (_dupenv_s(&buf, &sz, varname.data()) != 0 || buf == nullptr) {
        throw std::runtime_error("Failed to get environment variable: " + std::string(varname));
    }
    std::string result(buf);
    free(buf);
    #elif defined __linux__
    std::string result = std::getenv(varname.c_str());
    #endif
    return result;
}

void setupEnv() {
    // Disable OpenCV logging
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // Set GStreamer environment variables
    std::string vcpkg_root(getEnvironmentVariable("VCPKG_ROOT")), gstreamer_plugins, gstreamer_bin;
    std::filesystem::path gstreamer_plugins_path, gstreamer_bin_path;

    #ifdef _WIN32
    std::string platform = "x64-windows";
    std::string system_lib = getEnvironmentVariable("PATH");
    #elif defined __linux__
    std::string platform = "x64-linux-dynamic";
    std::string system_lib = getEnvironmentVariable("LD_LIBRARY_PATH");
    #endif

    #ifdef IS_DEBUG
        std::cout << cv::getBuildInformation() << std::endl;
        #ifdef WIN32
        _putenv("GST_DEBUG=3");
        #elif defined __linux__
        setenv("GST_DEBUG", "3", 1);
        #endif
        gstreamer_plugins_path = std::filesystem::path(vcpkg_root) / "installed" / platform / "debug" / "plugins" / "gstreamer";
        gstreamer_bin_path = std::filesystem::path(vcpkg_root) / "installed" / platform / "debug" / "bin";
    #else
        gstreamer_plugins_path = std::filesystem::path(vcpkg_root) / "installed" / platform / "plugins" / "gstreamer";
        gstreamer_bin_path = std::filesystem::path(vcpkg_root) / "installed" / platform / "bin";
    #endif
    gstreamer_plugins = gstreamer_plugins_path.string();
    gstreamer_bin = gstreamer_bin_path.string() + ";" + system_lib;
    #ifdef WIN32
    _putenv(("GST_PLUGIN_PATH=" + gstreamer_plugins).c_str());
    _putenv(("PATH=" + gstreamer_bin).c_str());
    #elif defined __linux__
    setenv("GST_PLUGIN_PATH", gstreamer_plugins_path.string().c_str(), 1);
    setenv("LD_LIBRARY_PATH", gstreamer_bin_path.string().c_str(), 1);
    #endif
}


int main(int argc, char** argv) {
    setupEnv();
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <gstInputPipeline> <gstOutputPipeline> "
                  << std::endl;
        return -1;
    }
    std::string gstInputPipeline(argv[1]),gstOutputPipeline(argv[2]);
    std::cout << "gstInputPipeline: " << gstInputPipeline << std::endl;
    std::cout << "gstOutputPipeline: " << gstOutputPipeline << std::endl;

    cv::VideoCapture cap(gstInputPipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cout << "Failed to open RTSP stream! Check gst input pipeline." << gstInputPipeline << std::endl;
        return -1;
    }

    int width   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps     = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    std::string outputCaps = cv::format("video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1",
                                        width, height, fps);
    std::string fullOutputPipeline = "appsrc is-live=true format=time do-timestamp=true ! " +
                                     outputCaps + " ! " +
                                     gstOutputPipeline;
    std::cout << "Output Pipeline: " << fullOutputPipeline << std::endl;

    cv::VideoWriter writer(fullOutputPipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);

    if (!writer.isOpened()) {
        std::cerr << "Failed to open VideoWriter! Check gst output pipeline: " << fullOutputPipeline << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Empty frame! Stop streaming." << std::endl;
            break;
        }
        writer.write(frame);
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}