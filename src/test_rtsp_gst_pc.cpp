#include <iostream>
#include <atomic>
#include <thread>

#include "utils.hpp"

// GSTREAMER 取流和推流，带有生产者-消费者模型
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
    double fps  = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Stream size: " << width << "x" << height << "@" << fps << "FPS" << std::endl;

    std::string outputCaps = cv::format("video/x-raw,format=BGR,width=%d,height=%d",
                                        width, height);
    std::string fullOutputPipeline = "appsrc is-live=true format=time do-timestamp=true ! " +
                                     outputCaps + " ! " +
                                     gstOutputPipeline;
    std::cout << "Output Pipeline: " << fullOutputPipeline << std::endl;

    cv::VideoWriter writer(fullOutputPipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);

    if (!writer.isOpened()) {
        std::cerr << "Failed to open VideoWriter! Check gst output pipeline: " << fullOutputPipeline << std::endl;
        return -1;
    }
    
    // 初始化生产者-消费者模型
    const size_t MAX_QUEUE_SIZE = 30;   // 缓冲区最大帧数
    FrameQueue<cv::Mat> frameBuffer(MAX_QUEUE_SIZE, "Buffer_Producer->Consumer");
    std::atomic<bool> isRunning(true);  // 原子变量：状态标志

    // 先启动生产者线程，负责读取帧
    std::thread producer_thread([&]() {
        std::cout << "[Producer] thread started." << std::endl;
        try {
            cv::Mat frame;
            while (isRunning) {
                cap >> frame;
                if (frame.empty()) {
                    std::cout << "[Producer] Empty frame! Stop streaming. Press Enter to quit." << std::endl;
                    break;
                }
                frameBuffer.push(frame);
            }
        } catch (const cv::Exception& e) {
            std::cerr << "[Producer] OpenCV Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Producer] Exception: " << e.what() << std::endl;
        }
        isRunning = false;
        frameBuffer.stop();
        std::cout << "[Producer] thread finished." << std::endl;
    });

    // 再启动消费者线程，负责写入帧
    std::thread consumer_thread([&]() {
        std::cout << "[Consumer] thread started." << std::endl;
        try {
            while (isRunning || !frameBuffer.empty()) {
                // 未按下停止键且视频流未结束 或 队列非空
                if (auto frame = frameBuffer.pop()) {
                    writer.write(*frame);
                } else {
                    // pop 返回false，意味着队列为空且已收到停止信号
                    break;
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "[Consumer] OpenCV Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Consumer] Exception: " << e.what() << std::endl;
        }
        std::cout << "[Consumer] thread finished." << std::endl;
    });

    // 最后主线程等待用户输入以退出
    std::cout << ">> Streaming... Press Enter to quit." << std::endl;
    std::cin.get();
    std::cout << ">> Stopping..." << std::endl;
    isRunning = false;
    frameBuffer.stop();
    if (producer_thread.joinable()) {
        producer_thread.join();
    }
    if (consumer_thread.joinable()) {
        consumer_thread.join();
    }
    writer.release();
    cap.release();
    return 0;
}