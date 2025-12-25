#include <iostream>
#include <atomic>
#include <thread>

#include "other_yolo.hpp"

// GSTREAMER 取流、YOLO模型推理、GSTREAMER 推流，采用三缓冲区生产者-消费者模型
// 缓冲区1: 取流 -> 推理 (原始帧)
// 缓冲区2: 推理 -> 绘图 (帧 + 推理结果)
// 缓冲区3: 绘图 -> 推流 (绘制后的帧)
int main(int argc, char** argv) {
    setupEnv();
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <gstInputPipeline> <gstOutputPipeline> <trtModelType> <trtModelPath>"
                  << std::endl;
        return -1;
    }
    std::string gstInputPipeline(argv[1]), gstOutputPipeline(argv[2]);
    std::cout << "gstInputPipeline: " << gstInputPipeline << std::endl;
    std::cout << "gstOutputPipeline: " << gstOutputPipeline << std::endl;

    std::string model_type(argv[3]), model_path(argv[4]);
    // auto model_pair = omodel_parse(model_type, model_path);
    // auto& model_variant = model_pair.first;
    // if (std::holds_alternative<std::monostate>(model_variant)) {
    //     return -1;
    // }
    // auto& model_params = model_pair.second;
    trtyolo::InferOption option;
    option.enableSwapRB();  // BGR->RGB转换
    trtyolo::SegmentModel model(model_path, option);
    auto model_params = segParams;
    const size_t batch_size = static_cast<size_t>(model.batch());

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

    // 初始化三缓冲区生产者-消费者模型
    const size_t MAX_QUEUE_SIZE = 30;   // 缓冲区最大帧数
    FrameQueue<OPreprocessedImage> frameBuffer_1(MAX_QUEUE_SIZE, "Buffer1_Grab->Infer");
    FrameQueue<OInferenceResult> frameBuffer_2(MAX_QUEUE_SIZE, "Buffer2_Infer->Plot");
    FrameQueue<cv::Mat> frameBuffer_3(MAX_QUEUE_SIZE, "Buffer3_Plot->Push");
    std::atomic<bool> isRunning(true);  // 原子变量：状态标志

    // 线程1: 取流线程，负责读取帧到 frameBuffer_1
    std::thread grab_thread([&]() {
        std::cout << "[GRAB] thread started." << std::endl;
        try {
            cv::Mat frame;
            while (isRunning) {
                cap >> frame;
                if (frame.empty()) {
                    std::cout << "[GRAB] Empty frame! Stop streaming." << std::endl;
                    break;
                }
                trtyolo::Image frame_data(
                    frame.data,  // 像素数据指针
                    frame.cols,  // 图像宽度
                    frame.rows   // 图像高度
                );
                frameBuffer_1.push(OPreprocessedImage{frame.clone(), frame_data});
            }
        } catch (const cv::Exception& e) {
            std::cerr << "[GRAB] OpenCV Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[GRAB] Exception: " << e.what() << std::endl;
        }
        isRunning = false;
        frameBuffer_1.stop();
        std::cout << "[GRAB] thread finished." << std::endl;
    });

    // 线程2: 推理线程，负责从 frameBuffer_1 取帧，进行推理后将 <frame, result> 放入 frameBuffer_2
    std::thread infer_thread([&]() {
        std::cout << "[INFER] thread started." << std::endl;
        std::vector<OPreprocessedImage> batch_images;
        std::vector<trtyolo::Image> datas;
        batch_images.reserve(batch_size);
        datas.reserve(batch_size);
        try {
            auto process_batch = [&]() {
                if (batch_images.empty()) return;
                auto results = model.predict(datas);
                for (size_t i = 0; i < results.size(); i++) {
                    frameBuffer_2.push(OInferenceResult{
                        .image = batch_images[i].image.clone(),
                        .result = results[i]
                    });
                }
                batch_images.clear();
                datas.clear();
            };

            while (isRunning || !frameBuffer_1.empty()) {
                if (auto frame = frameBuffer_1.pop()) {
                    // 先将当前帧加入批次
                    batch_images.emplace_back(*frame);
                    datas.emplace_back(frame->data);

                    // 如果批次已满，进行推理
                    if (batch_images.size() >= batch_size) {
                        process_batch();
                    }
                } else {
                    // 队列已停止或无数据，处理剩余部分批次后退出
                    process_batch();
                    break;
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "[INFER] OpenCV Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[INFER] Exception: " << e.what() << std::endl;
        }
        frameBuffer_2.stop();
        std::cout << "[INFER] thread finished." << std::endl;
    });

    // 线程3: 绘图线程，负责从 frameBuffer_2 取帧和推理结果，绘图后放入 frameBuffer_3
    std::thread plot_thread([&]() {
        std::cout << "[PLOT] thread started." << std::endl;
        try {
            while (isRunning || !frameBuffer_2.empty()) {
                if (auto infer_result = frameBuffer_2.pop()) {
                    // 绘制推理结果
                    visualize(infer_result->image, infer_result->result, model_params.class_names);
                    // 将绘制后的帧放入 frameBuffer_3
                    frameBuffer_3.push(infer_result->image);
                } else {
                    break;
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "[PLOT] OpenCV Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[PLOT] Exception: " << e.what() << std::endl;
        }
        frameBuffer_3.stop();
        std::cout << "[PLOT] thread finished." << std::endl;
    });

    // 线程4: 推流线程，负责从 frameBuffer_3 取帧推流
    std::thread push_thread([&]() {
        std::cout << "[PUSH] thread started." << std::endl;
        try {
            while (isRunning || !frameBuffer_3.empty()) {
                if (auto frame = frameBuffer_3.pop()) {
                    writer.write(*frame);
                } else {
                    break;
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "[PUSH] OpenCV Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[PUSH] Exception: " << e.what() << std::endl;
        }
        std::cout << "[PUSH] thread finished." << std::endl;
    });

    // 主线程等待用户输入以退出
    std::cout << ">> Streaming... Press Enter to quit." << std::endl;
    std::cin.get();
    std::cout << ">> Stopping..." << std::endl;
    isRunning = false;
    frameBuffer_1.stop();
    frameBuffer_2.stop();
    frameBuffer_3.stop();

    if (grab_thread.joinable()) {
        grab_thread.join();
    }
    if (infer_thread.joinable()) {
        infer_thread.join();
    }
    if (plot_thread.joinable()) {
        plot_thread.join();
    }
    if (push_thread.joinable()) {
        push_thread.join();
    }
    writer.release();
    cap.release();
    return 0;
}
