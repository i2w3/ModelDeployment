#include <iostream>
#include <atomic>
#include <thread>

#include "utils.hpp"


// GSTREAMER 取流、YOLO TRT 推理和推流
int main(int argc, char** argv) {
    setupEnv();
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <gstInputPipeline> <gstOutputPipeline> <trtModelType> <trtModelPath>"
                  << std::endl;
        return -1;
    }
    std::string gstInputPipeline(argv[1]),gstOutputPipeline(argv[2]);
    std::cout << "gstInputPipeline: " << gstInputPipeline << std::endl;
    std::cout << "gstOutputPipeline: " << gstOutputPipeline << std::endl;

    std::string model_type(argv[3]), model_path(argv[4]);
    auto model_pair = model_parse(model_type, model_path);
    if (!model_pair.first) {
        return -1;
    }
    auto& model = model_pair.first;
    auto& model_params = model_pair.second;

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

    cv::VideoWriter writer(fullOutputPipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);
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
            int frame_count = 0;
            auto last_time = std::chrono::steady_clock::now();
            double current_fps = 0.0;
            std::string fps_text;
            cv::Point fps_pos = cv::Point(20, 40);
            while (isRunning) {
                cap >> frame;
                if (frame.empty()) {
                    std::cout << "Empty frame! Stop streaming. Press Enter to quit." << std::endl;
                    break;
                }
                auto result = model->infer(frame);
                plot_result(frame, result, model_params);
                frame_count++;
                auto current_time = std::chrono::steady_clock::now();
                // 计算自上次更新以来经过的时间（秒）
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_time).count();
                // 如果超过 1 秒，就更新一次 FPS 显示
                if (elapsed_seconds >= 1) {
                    if (elapsed_seconds > 0) { // 避免除以零
                        current_fps = frame_count / static_cast<double>(elapsed_seconds);
                    }
                    fps_text = cv::format("FPS: %.2f", current_fps);
                    // 重置计数器和时间戳
                    frame_count = 0;
                    last_time = current_time;
                }
                // 显示 FPS
                cv::putText(frame, fps_text, fps_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 4);
                cv::putText(frame, fps_text, fps_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
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