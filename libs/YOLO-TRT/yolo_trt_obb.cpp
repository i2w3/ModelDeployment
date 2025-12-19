#include "yolo_trt.h"

std::vector<YOLODetection> YOLOObb::postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) {
    std::vector<YOLODetection> detections;
    auto output = outputs.at("output0");

    // 获取输出维度: [1, 4 + num_classes + rad, num_boxes]
    // 4 表示: center_x, center_y, width, height
    const int num_boxes = output.size[2]; // num_boxes

    // 重塑并转置输出矩阵为 [num_boxes, 4 + num_classes + rad] 格式，方便遍历检测框
    cv::Mat output_mat = output.reshape(1, output.size[1]);
    output_mat = output_mat.t();

    // 预分配 vector 以避免频繁的内存重新分配
    std::vector<cv::RotatedRect> rotated_boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    rotated_boxes.reserve(num_boxes);
    confidences.reserve(num_boxes);
    class_ids.reserve(num_boxes);

    // 处理每个检测框
    for (int i = 0; i < num_boxes; i++) {
        cv::Mat row = output_mat.row(i);
        float* data = row.ptr<float>();

        // 提取边界框坐标 (中心格式: xc, yc, w, h)
        float xc = data[0];  // 中心x坐标
        float yc = data[1];  // 中心y坐标
        float w = data[2];   // 宽度
        float h = data[3];   // 高度
        float rad = data[4 + this->modelParams.num_classes]; // 弧度
        float angle = rad * 180.0f / static_cast<float>(CV_PI); // 转换为角度

        // 提取类别置信度分数 (此检测的所有类别)
        cv::Mat scores = row.colRange(4, 4 + this->modelParams.num_classes);

        // 找到置信度最高的类别
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id_point);

        // 过滤掉低于置信度阈值的检测
        if (max_class_score > this->modelParams.conf_threshold) {
            rotated_boxes.emplace_back(cv::RotatedRect(cv::Point2f(xc, yc), cv::Size2f(w, h), angle));
            confidences.emplace_back(static_cast<float>(max_class_score));
            class_ids.emplace_back(class_id_point.x);  // class_id_point.x 包含类别索引
        }
    }

    // 应用非极大值抑制(NMS)消除冗余的重叠框
    std::vector<int> indices;
    if (!rotated_boxes.empty()) {
        cv::dnn::NMSBoxes(rotated_boxes, confidences, 0.0f, this->modelParams.nms_threshold, indices, 1.f, this->modelParams.max_det);
    }

    // 将过滤后的索引转换为最终的YOLODetection对象
    detections.reserve(indices.size());  // 预分配最终结果vector
    for (int idx : indices) {
        YOLODetection det;
        det.rotatedBox = rotated_boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }
    for (auto& det : detections) {
        // 调整边界框以补偿前处理中的缩放和填充
        det.rotatedBox.center.x = static_cast<float>((det.rotatedBox.center.x - params.left) / params.scale);
        det.rotatedBox.center.y = static_cast<float>((det.rotatedBox.center.y - params.top) / params.scale);
        det.rotatedBox.size.width = static_cast<float>(det.rotatedBox.size.width / params.scale);
        det.rotatedBox.size.height = static_cast<float>(det.rotatedBox.size.height / params.scale);
    }
    return detections;
}