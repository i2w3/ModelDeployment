#include <iostream>
#include <atomic>
#include <thread>

#include "utils.hpp"

// FFMPEG 取流和 CLI 推流：VLC 启动较慢，延迟约 1.5s
int main(int argc, char** argv) {
    setupEnv();
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <rtspInputUrl> <rtspOutputUrl> \n"
                  << "Example: for windows: rtsp://127.0.0.1:8554/live rtsp://127.0.0.1:8554/output \n"
                  << "         for linux:   rtsp://mediamtx:8554/live  rtsp://mediamtx:8554/output \n"
                  << std::endl;
        return -1;
    }
    std::string rtspInputUrl(argv[1]),rtspOutputUrl(argv[2]);
    std::cout << "RTSP Input URL: " << rtspInputUrl << std::endl;
    std::cout << "RTSP Output URL: " << rtspOutputUrl << std::endl;
    
    // init VideoCapture with FFMPEG backend
    cv::VideoCapture cap(rtspInputUrl, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cout << "Failed to open RTSP stream!" << std::endl;
        return -1;
    }
    
    int width   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    float fps   = static_cast<float>(cap.get(cv::CAP_PROP_FPS));
    // init FFMPEG pipe for RTSP output
    std::string ffmpegCMD;
    std::filesystem::path ffmpegPath;
#ifdef WIN32
    std::cout << "Running on Windows" << std::endl;
    ffmpegPath = "ffmpeg"; // 假设已经添加进 PATH
#elif defined __linux__
    std::cout << "Running on Linux" << std::endl;
    // Use the FFmpeg installed by vcpkg: vcpkg install ffmpeg[ffmepg]
    const char* vcpkgRootRaw = std::getenv("VCPKG_ROOT");
    if (vcpkgRootRaw == nullptr) {
        std::cerr << "Error! VCPKG_ROOT environment variable is not set." << std::endl;
        return -1;
    }
    std::filesystem::path vcpkgPath = vcpkgRootRaw;
    ffmpegPath = vcpkgPath / "installed" / "x64-linux-dynamic" / "tools" / "ffmpeg" / "ffmpeg";
    if (!std::filesystem::exists(ffmpegPath)) {
        std::cerr << "Error! Can't find " << ffmpegPath << std::endl;
        return -1;
    }
#endif
    ffmpegCMD = ffmpegPath.string() + " -f rawvideo -pixel_format bgr24" + 
                " -video_size " + std::to_string(width) + "x" + std::to_string(height) +
                " -framerate " + std::to_string(fps) +
                " -i - -c:v h264_nvenc -an" + // 设置编码器
                " -g 30 -bf 0 -tune ull -preset p3 -rc cbr -b:v 8M -profile:v high"
                " -rtsp_transport tcp -f rtsp " + rtspOutputUrl;
    std::cout << "FFmpeg command: " << ffmpegCMD << std::endl;

#ifdef WIN32
    FILE* ffmpegPipe = _popen(ffmpegCMD.c_str(), "wb");
#elif defined __linux__
    FILE* ffmpegPipe = popen(ffmpegCMD.c_str(), "w");
#endif
    std::cout << "FFmpeg RTSP Output Pipe initialized." << std::endl;

    std::atomic<bool> isRunning(true);  // 原子变量：状态标志
    // 启动线程，方便接收键盘输入
    std::thread task_thread([&]() {
    std::cout << "[thread] started." << std::endl;
    try {
        cv::Mat frame;
        while (isRunning) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << "Empty frame! Stop streaming. Press Enter to quit." << std::endl;
                break;
            }
            fwrite(frame.data, 1, frame.total() * frame.elemSize(), ffmpegPipe);
            fflush(ffmpegPipe);
            }
        } catch (const cv::Exception& e) {
            std::cerr << "[thread] OpenCV Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[thread] Exception: " << e.what() << std::endl;
        }
        std::cout << "[thread] thread finished." << std::endl;
    });

    std::cout << ">> Streaming... Press Enter to quit." << std::endl;
    std::cin.get();
    std::cout << ">> Stopping..." << std::endl;
    isRunning = false;
    if (task_thread.joinable()) {
        task_thread.join();
    }

#ifdef WIN32
    _pclose(ffmpegPipe);
#elif defined __linux__
    pclose(ffmpegPipe);
#endif
    cap.release();
    return 0;
}