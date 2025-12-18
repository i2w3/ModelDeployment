#include <iostream>
#include <filesystem>
#include <cstdlib> // _popen and _pclose
#include <opencv2/opencv.hpp>

// FFMPEG 取流和 CLI 推流（注意添加 FFMPEG 到系统环境变量中去，仅在 Windows 上测试通过）
// ffmpeg -re -i "C:\Users\blxtw\Desktop\vp_data\test_video\face.mp4" -c:v libx264 -g 15 -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live

void setupEnv() {
    // Disable OpenCV logging
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
}


int main() {
    setupEnv();
    std::string rtspInputUrl, ffmpegCmd;
    // std::cout << "Enter the url to open the RTSP stream: ";
    // std::cin >> rtspInputUrl;
    rtspInputUrl = "rtsp://localhost:8554/live";
    cv::VideoCapture cap(rtspInputUrl, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cout << "Failed to open RTSP stream!" << std::endl;
        return -1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    ffmpegCmd = "ffmpeg -y -f rawvideo -pixel_format bgr24 -video_size " + 
                std::to_string(width) + "x" + std::to_string(height) + 
                " -i - -c:v libx264 -preset ultrafast -tune zerolatency " + 
                " -g 30 -f rtsp rtsp://127.0.0.1:8554/output";
    FILE* ffmpegPipe = _popen(ffmpegCmd.c_str(), "wb");
    if (!ffmpegPipe) {
        std::cerr << "Could not open pipe to FFmpeg" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Empty frame" << std::endl;
            break;
        }
        // cv::imshow("RTSP Stream", frame);
        fwrite(frame.data, 1, frame.total() * frame.elemSize(), ffmpegPipe);
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }
    _pclose(ffmpegPipe);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}