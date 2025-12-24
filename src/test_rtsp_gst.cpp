#include <iostream>
#include <atomic>
#include <thread>

#include "utils.hpp"

// GSTREAMER 取流和推流
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

    // init VideoCapture with GSTREAMER backend
    cv::VideoCapture cap(gstInputPipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "Failed to open RTSP stream! Check gst input pipeline." << gstInputPipeline << std::endl;
        return -1;
    }

    int width   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps  = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Stream size: " << width << "x" << height << "@" << fps << "FPS" << std::endl;

    std::string outputCaps = cv::format("video/x-raw,format=BGR,width=%d,height=%d",
                                        width, height);
    std::string fullOutputPipeline = "appsrc is-live=true format=time do-timestamp=true ! " +
                                     outputCaps + " ! " +
                                     gstOutputPipeline;
    std::cout << "Output Pipeline: " << fullOutputPipeline << std::endl;

    cv::VideoWriter writer(fullOutputPipeline, cv::CAP_GSTREAMER, 0, std::round(fps), cv::Size(width, height), true);
    if (!writer.isOpened()) {
        std::cerr << "Failed to open VideoWriter! Check gst output pipeline: " << fullOutputPipeline << std::endl;
        return -1;
    }

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
                writer.write(frame);
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
    writer.release();
    cap.release();
    return 0;
}