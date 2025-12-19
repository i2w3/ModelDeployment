#include "yolo_trt.h"

std::vector<YOLODetection> YOLOSeg::postprocess(const std::unordered_map<std::string, cv::Mat>& outputs, const PreprocessParams& params) {
    std::vector<YOLODetection> detections;
    auto preds = outputs.at("output0"); // [1, 4 + num_classes + 32, num_boxes]
    auto proto = outputs.at("output1"); // [1, 32, proto_h, proto_w]

    // 获取 preds 输出维度: [1, 4 + num_classes + 32, num_boxes]
    // 4 表示: center_x, center_y, width, height
    const int num_boxes = preds.size[2]; // num_boxes

    // 重塑并转置输出矩阵为 [num_boxes, 4 + num_classes] 格式，方便遍历检测框
    cv::Mat output_mat = preds.reshape(1, preds.size[1]);
    output_mat = output_mat.t();

    // 获取 proto 输出维度： [1, 32, proto_h, proto_w]
    const int proto_h = proto.size[2];
    const int proto_w = proto.size[3];
    const int proto_area = proto_h * proto_w;

    // 直接从 proto 数据构造protos矩阵：[32, proto_h * proto_w]
    cv::Mat protos = cv::Mat(32, proto_area, CV_32F, proto.ptr<float>());

    // 预分配 vector 以避免频繁的内存重新分配
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    boxes.reserve(num_boxes);
    confidences.reserve(num_boxes);
    class_ids.reserve(num_boxes);

    // 保存过滤后的 mask coefficients
    cv::Mat masks_coeffs; 

    // 处理每个检测框
    for (int i = 0; i < num_boxes; i++) {
        cv::Mat row = output_mat.row(i);
        float* data = row.ptr<float>();

        // 提取边界框坐标 (中心格式: xc, yc, w, h)
        float xc = data[0];  // 中心x坐标
        float yc = data[1];  // 中心y坐标
        float w = data[2];   // 宽度
        float h = data[3];   // 高度

        // 提取类别置信度分数 (此检测的所有类别)
        cv::Mat scores = row.colRange(4, 4 + this->modelParams.num_classes);

        // 找到置信度最高的类别
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id_point);

        // 过滤掉低于置信度阈值的检测
        if (max_class_score > this->modelParams.conf_threshold) {
            // 从中心坐标转换为左上角坐标
            float x = xc - w / 2.0f;  // 左边界
            float y = yc - h / 2.0f;  // 上边界

            // 保存 mask conefficients
            cv::Mat mask_coeff = cv::Mat(1, 32, CV_32F, data + 4 + this->modelParams.num_classes);
            
            // 将检测结果存储在临时 vector 中以供NMS处理
            boxes.emplace_back(cv::Rect2f(x,y,w,h));
            confidences.emplace_back(static_cast<float>(max_class_score));
            class_ids.emplace_back(class_id_point.x);  // class_id_point.x 包含类别索引

            // 直接将 mask_coeff 添加到 masks_coeffs 矩阵中
            masks_coeffs.push_back(mask_coeff);
        }
    }

    // 应用非极大值抑制(NMS)消除冗余的重叠框
    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confidences, 0.0f, this->modelParams.nms_threshold, indices, 1.f, this->modelParams.max_det);
    }


    detections.reserve(indices.size()); // 预分配最终结果vector
    cv::Mat filtered_masks_coeffs;      // 保存NMS后对应的mask系数

    for (int idx : indices) {
        YOLODetection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
        filtered_masks_coeffs.push_back(masks_coeffs.row(idx));
    }

    // 计算每个检测框的分割掩码
    cv::Mat matmulRes = (filtered_masks_coeffs * protos).t();
    cv::Mat maskMat = matmulRes.reshape(static_cast<int>(indices.size()), {proto_w, proto_h}); // [num_boxes, proto_h, proto_w]

    std::vector<cv::Mat> maskChannels;
    maskChannels.reserve(indices.size());
    cv::split(maskMat, maskChannels);

    // 跳过对 letterboc 区域的掩码计算
    cv::Rect roi;
    if (params.top == 0) {
        // 水平方向有 padding
        int roi_x = static_cast<int>(params.left / static_cast<float>(this->modelParams.image_size) * proto_w);
        int roi_y = 0;
        int roi_w = static_cast<int>((this->modelParams.image_size - params.left - params.right) / static_cast<float>(this->modelParams.image_size) * proto_w);
        int roi_h = proto_h;
        roi = cv::Rect(roi_x, roi_y, roi_w, roi_h);
    } else if (params.left == 0) {
        // 垂直方向有 padding
        int roi_x = 0;
        int roi_y = static_cast<int>(params.top / static_cast<float>(this->modelParams.image_size) * proto_h);
        int roi_w = proto_w;
        int roi_h = static_cast<int>((this->modelParams.image_size - params.top - params.bottom) / static_cast<float>(this->modelParams.image_size) * proto_h);
        roi = cv::Rect(roi_x, roi_y, roi_w, roi_h);
    } else {
        std::cerr << "Both top and left padding are non-zero, which is unexpected." << std::endl;
    }
    std::cout << "ROI for mask cropping: " << roi << std::endl;

    for (size_t i = 0; i < detections.size(); i++) {
        cv::Mat dest, mask;
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest); // [proto_h, proto_w]
        dest = dest(roi); // 裁剪掉 letterbox 区域
        cv::resize(dest, mask, cv::Size(static_cast<int>(params.original.cols), static_cast<int>(params.original.rows)), cv::INTER_LINEAR); // 调整掩码大小以匹配原始图像尺寸
        std::cout << "Mask size after resize: " << mask.size() << std::endl;
        // 调整边界框以补偿前处理中的缩放和填充
        detections[i].box.x = static_cast<int>((detections[i].box.x - params.left) / params.scale);
        detections[i].box.y = static_cast<int>((detections[i].box.y - params.top) / params.scale);
        detections[i].box.width = static_cast<int>(detections[i].box.width / params.scale);
        detections[i].box.height = static_cast<int>(detections[i].box.height / params.scale);
        detections[i].boxMask = mask(detections[i].box) > this->modelParams.seg_threshold;
    }
    return detections;
}
