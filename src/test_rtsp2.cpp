#include <iostream>
#include <filesystem>
#include <cstdlib> // _popen and _pclose
#include <opencv2/opencv.hpp>
#include <filesystem>

// FFMPEG 取流和 CLI 推流，目测目前延迟在 2s
void setupEnv() {
    // Disable OpenCV logging
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
}

int main(int argc, char** argv) {
    setupEnv();
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <rtsp_url> \n"
                  << "Example: for windows: rtsp://127.0.0.1:8554/live \n"
                  << "         for linux:   rtsp://mediamtx:8554/live"
                  << std::endl;
        return -1;
    }
    std::string rtspInputUrl = argv[1];
    std::cout << "RTSP Input URL: " << rtspInputUrl << std::endl;
    // init VideoCapture with FFMPEG backend
    cv::VideoCapture cap(rtspInputUrl, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cout << "Failed to open RTSP stream!" << std::endl;
        return -1;
    }
    int width   = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height  = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    float fps   = (float)cap.get(cv::CAP_PROP_FPS);
    // init FFMPEG pipe for RTSP output
    std::string ffmpegCMD;
    #ifdef WIN32
    std::cout << "Running on Windows" << std::endl;
    ffmpegCMD = "ffmpeg -f rawvideo -pixel_format bgr24"
                " -video_size " + std::to_string(width) + "x" + std::to_string(height) +
                " -framerate " + std::to_string(fps) +
                " -i - -c:v h264_nvenc -an" + // 设置编码器
                " -preset p3" + // 优化项
                " -rtsp_transport tcp -f rtsp rtsp://127.0.0.1:8554/output";
    std::cout << "FFmpeg command: " << ffmpegCMD << std::endl;
    FILE* ffmpegPipe = _popen(ffmpegCMD.c_str(), "wb");
    #elif defined __linux__
    std::cout << "Running on Linux" << std::endl;
    // Use the FFmpeg installed by vcpkg: vcpkg install ffmpeg[ffmepg]
    const char* vcpkgRootRaw = std::getenv("VCPKG_ROOT");
    if (vcpkgRootRaw == nullptr) {
        std::cerr << "Error! VCPKG_ROOT environment variable is not set." << std::endl;
        return -1;
    }
    std::filesystem::path vcpkgPath = vcpkgRootRaw;
    std::filesystem::path ffmpegPath = vcpkgPath / "installed" / "x64-linux-dynamic" / "tools" / "ffmpeg" / "ffmpeg";
    if (!std::filesystem::exists(ffmpegPath)) {
        std::cerr << "Error! Can't find " << ffmpegPath << std::endl;
        return -1;
    }

    ffmpegCMD = ffmpegPath.string() + " -f rawvideo -pixel_format bgr24" + 
                " -video_size " + std::to_string(width) + "x" + std::to_string(height) +
                " -framerate " + std::to_string(fps) +
                " -i - -c:v h264_nvenc -an" + // 设置编码器
                " -preset p3" + // 优化项
                " -rtsp_transport tcp -f rtsp rtsp://mediamtx:8554/output";
    std::cout << "FFmpeg command: " << ffmpegCMD << std::endl;
    FILE* ffmpegPipe = popen(ffmpegCMD.c_str(), "w");
    #endif
    std::cout << "FFmpeg RTSP Output Pipe initialized." << std::endl;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Empty frame" << std::endl;
            break;
        }
        fwrite(frame.data, 1, frame.total() * frame.elemSize(), ffmpegPipe);
    }

    #ifdef WIN32
    _pclose(ffmpegPipe);
    #elif defined __linux__
    pclose(ffmpegPipe);
    #endif
    
    cap.release();
    return 0;
}