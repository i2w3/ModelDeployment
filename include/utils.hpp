#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <filesystem>
#include <cstdlib>
#include <condition_variable>
#include <queue>
#include <utility>
#include <optional>

#include <opencv2/opencv.hpp>

#include "yolo_trt.h"

// 获取环境变量的值
inline std::string getEnvironmentVariable(const std::string& varname) {
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

// 环境配置：禁用 OpenCV 日志，设置 GStreamer 路径
inline void setupEnv() {
    // Disable OpenCV logging
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // Set GStreamer environment variables
    std::string gstreamer_plugins, gstreamer_bin, vcpkg_root(getEnvironmentVariable("VCPKG_ROOT"));
    std::filesystem::path gstreamer_plugins_path, gstreamer_bin_path, gstreamer_tools_path;

#ifdef _WIN32
    std::string platform = "x64-windows";
    std::string system_lib = getEnvironmentVariable("PATH");
#elif defined __linux__
    std::string platform = "x64-linux-dynamic";
    std::string system_lib = getEnvironmentVariable("LD_LIBRARY_PATH");
#endif
    gstreamer_tools_path = std::filesystem::path(vcpkg_root) / "installed" / platform / "tools" / "gstreamer";

#ifdef IS_DEBUG
    std::cout << cv::getBuildInformation() << std::endl;
    #ifdef WIN32
        _putenv("GST_DEBUG=3");
    #elif defined __linux__
        setenv("GST_DEBUG", "3", 1);
    #endif
#endif
    gstreamer_plugins_path = std::filesystem::path(vcpkg_root) / "installed" / platform / "plugins" / "gstreamer";
    gstreamer_bin_path = std::filesystem::path(vcpkg_root) / "installed" / platform / "bin";
    gstreamer_plugins = gstreamer_plugins_path.string();
    gstreamer_bin = gstreamer_bin_path.string() + ":" + system_lib;
#ifdef WIN32
    _putenv(("GST_PLUGIN_PATH=" + gstreamer_plugins).c_str());
    _putenv(("GST_PLUGIN_SCANNER=" + (gstreamer_tools_path / "gst-plugin-scanner.exe").string()).c_str());
    _putenv(("PATH=" + gstreamer_bin).c_str());
#elif defined __linux__
    setenv("GST_PLUGIN_PATH", gstreamer_plugins_path.string().c_str(), 1);
    setenv("GST_PLUGIN_SCANNER", (gstreamer_tools_path / "gst-plugin-scanner").string().c_str(), 1);
    setenv("LD_LIBRARY_PATH", gstreamer_bin_path.string().c_str(), 1);
#endif
}

// 模型 parse
inline std::pair<std::unique_ptr<YOLODetector>, ModelParams> model_parse(const std::string& model_type, const std::string& model_path) {
    // 定义在 yolo_trt.h 中
    extern const ModelParams clsParams, detParams, obbParams, poseParams, segParams;

    std::unique_ptr<YOLODetector> model;
    ModelParams model_params;

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
    else {
        std::cerr << "Unsupported model type!" << std::endl;
        return std::make_pair(nullptr, ModelParams{});
    }
    return std::make_pair(std::move(model), model_params);
}

// 存储帧和推理结果的配对
struct InferenceResult {
    cv::Mat frame;
    std::vector<YOLODetection> result;
};

// 深拷贝辅助函数：支持 cv::Mat 和普通类型
template<typename T>
T deep_copy(const T& val) {
    // 普通类型直接返回
    return val;
}

template<>
cv::Mat deep_copy<cv::Mat>(const cv::Mat& val) {
    // cv::Mat 特化：调用 clone() 深拷贝像素数据
    return val.clone();
}

template<>
InferenceResult deep_copy<InferenceResult>(const InferenceResult& val) {
    // InferenceResult 特化：深拷贝内部的 cv::Mat
    return InferenceResult{val.frame.clone(), val.result};
}


template <typename T>
class FrameQueue {
public:
    explicit FrameQueue(size_t _max_size, const std::string& name = "FrameQueue")
        : max_size(_max_size), stop_flag(false), queue_name(name) {}

    // 生产者调用：向队列中添加帧
    void push(const T& frame) {
        std::unique_lock<std::mutex> lock(this->mtx); // 上锁，禁止其它线程访问 this->queue
        bool was_full = false;
        cond_producer.wait(lock, [this, &was_full] {
            // 队列已满或者收到停止信息，返回 false，wait 自动释放 lock 并阻塞当前线程，等待 this->cond_producer.notify_one()
            // 队列未满或者未收到停止信息，返回 true，wait 重新获得锁 lock 并继续执行
            if (this->queue.size() >= this->max_size && !this->stop_flag) {
                was_full = true;
            }
            return this->queue.size() < this->max_size || this->stop_flag;
        });
        if (was_full) {
            std::cerr << "[" << queue_name << "] WARNING: Queue was full (" << max_size
                      << "), producer was blocked!" << std::endl;
        }
        if (this->stop_flag) {
            // 阻塞结束后检查是否收到停止信号，如果是则直接返回
            return;
        }
        // 对 cv::Mat 和 InferenceResult 进行深拷贝，避免多线程共享像素数据
        this->queue.push(deep_copy(frame));
        this->cond_consumer.notify_one();   // 通知等待的消费者线程有新数据可用
        // lock 开始析构自动解锁
    }

    // 消费者调用：从队列中取出帧
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(this->mtx); // 上锁，禁止其它线程访问 this->queue
        bool was_empty = false;
        this->cond_consumer.wait(lock, [this, &was_empty] {
            // 队列为空或者收到停止信息，返回 false，wait 自动释放 lock 并阻塞当前线程，等待 this->cond_consumer.notify_one()
            // 队列非空或者未收到停止信息，返回 true，wait 重新获得锁 lock 并继续执行
            if (this->queue.empty() && !this->stop_flag) {
                was_empty = true;
            }
            return !this->queue.empty() || this->stop_flag;
        });
    #ifdef IS_DEBUG
        if (was_empty) {
            std::cerr << "[" << queue_name << "] WARNING: Queue was empty, consumer was blocked!" << std::endl;
        }
    #endif
        if (this->queue.empty()) {
            // 阻塞结束后检查队列是否为空，即使收到停止信号也要等待处理完队列所有数据
            return std::nullopt;
        }
        T frame = std::move(this->queue.front());   // 取出队首元素
        this->queue.pop();                          // 弹出队首元素
        this->cond_producer.notify_one();           // 通知等待的生产者线程有空间可用
        return frame;
    }

    // 检查队列是否为空
    bool empty() const {
        std::lock_guard<std::mutex> lock(this->mtx);
        return this->queue.empty();
    }

    // 获取当前队列大小
    size_t size() const {
        std::lock_guard<std::mutex> lock(this->mtx);
        return this->queue.size();
    }

    // 通知所有等待的线程
    void stop() {
        std::unique_lock<std::mutex> lock(this->mtx);
        this->stop_flag = true;
        this->cond_producer.notify_all();
        this->cond_consumer.notify_all();
    }

private:
    std::queue<T> queue;
    mutable std::mutex mtx;
    std::condition_variable cond_producer;
    std::condition_variable cond_consumer;
    const size_t max_size;
    bool stop_flag;
    std::string queue_name;
};


#endif // UTILS_H