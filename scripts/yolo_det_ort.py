import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
CLASSES_NAMES  = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


@dataclass
class YOLODetInput:
    image: np.ndarray
    blob: np.ndarray
    top: int
    bottom: int
    left: int
    right: int
    scale: float


@dataclass
class YOLODetOutput:
    box: list[int] # [x1, y1, w, h]
    score: float
    class_id: int


def preProcess(src:np.ndarray, size:int) -> np.ndarray:
    # letterBox mode
    h, w = src.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized_img = cv2.resize(src, (nw, nh))

    padW = (size - nw) / 2.
    padH = (size - nh) / 2.

    top = int(padH - 0.1)
    bottom = int(padH + 0.1)
    left = int(padW - 0.1)
    right = int(padW + 0.1)

    boxed_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
    blob = cv2.dnn.blobFromImage(boxed_img, 1/255.0, (size, size), swapRB=True, crop=False, ddepth=cv2.CV_32F)
    return YOLODetInput(image=src, blob=blob, top=top, bottom=bottom, left=left, right=right, scale=scale)


def postProcess(outputs: list[np.ndarray]) -> None:
    assert len(outputs) == 1, "Expected a single output from the model."
    assert outputs[0].ndim == 3, "Expected output to be a 3D array."
    predictions = outputs[0][0]
    predictions = predictions.transpose(1, 0)

    class_ids = []
    boxes = []
    confidences = []

    for pred in predictions:
        score = pred[4:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > CONF_THRESHOLD:
            # 最好先做个置信度过滤，减少 NMS 计算量
            cx, cy, w, h = pred[0:4]
            x = int(cx - w / 2.0)
            y = int(cy - h / 2.0)
            boxes.append([x, y, int(w), int(h)])
            class_ids.append(class_id)
            confidences.append(float(confidence))
    
    # 注意不是按类别 nms，而是所有类别一起做 nms
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)
    indices = nms(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)
    results = []
    for i in indices:
        box = boxes[i]
        score = confidences[i]
        class_id = class_ids[i]
        results.append(YOLODetOutput(box=box, score=score, class_id=class_id))
    return results


def nms(boxes: list[list[int]], 
        scores: list[float], 
        conf_threshold:float, 
        iou_threshold: float) -> list[int]:
    # indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    # 1. 置信度过滤
    filtered_indices = [i for i, s in enumerate(scores) if s >= conf_threshold]
    if not filtered_indices:
        return []
    filtered_boxes = [boxes[i] for i in filtered_indices]
    filtered_scores = [scores[i] for i in filtered_indices]

    # 2. 按置信度排序
    sorted_idx = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)

    # 3. 逐个比较
    keep_local = []
    while sorted_idx:
        current = sorted_idx.pop(0)
        keep_local.append(current)

        remaining = []
        for idx in sorted_idx:
            if iou(filtered_boxes[current], filtered_boxes[idx]) <= iou_threshold:
                remaining.append(idx)
        sorted_idx = remaining
    indices = [filtered_indices[i] for i in keep_local]

    return indices


def iou(box1:list[int], box2:list[int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area
    

if __name__ == "__main__":
    # setting ort environment
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # 启用所有优化
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024, # 4 GB
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './res/yolo/trt_cache',
            'trt_timing_cache_enable': True, # timing cache 加速在其它设备上建立 engine
            'trt_timing_cache_path': './res/yolo/trt_cache/time_cache',
            'trt_force_timing_cache': False, # 仅在与生成 timing cache 的 GPU 型号完全相同的 GPU 上使用
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })
    ]

    # Load ONNX model
    model_path = "res/yolo/yolo11n.onnx"
    start_time = time.time()
    session = ort.InferenceSession(model_path, 
                                   sess_options=so,
                                   providers=providers)
    end_time = time.time()
    load_time_ms = (end_time - start_time) * 1000
    print(f"Model loaded in {load_time_ms:.2f} ms.")

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    print(f"Input name: {input_name}, shape: {input_shape}")
    print(f"Output name: {output_name}, shape: {output_shape}")

    # Warm up
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    session.run(None, {session.get_inputs()[0].name: dummy_input})

    # Run inference for {runs} times to measure performance
    image_path = "res/bus.jpg"
    runs = 1000
    start_time = time.time()
    for _ in range(runs):
        image = cv2.imread(image_path)
        input_size = (640, 640)
        det_input = preProcess(image, input_size[0])
        outputs = session.run([output_name], {input_name: det_input.blob})
        results = postProcess(outputs)
    end_time = time.time()
    avg_time_ms = (end_time - start_time) / runs * 1000
    print(f"Average inference time over {runs} runs: {avg_time_ms:.2f} ms")

    # Draw results on the image
    for det in results:
        x, y, w, h = det.box
        x = int((x - det_input.left) / det_input.scale)
        y = int((y - det_input.top) / det_input.scale)
        w = int(w / det_input.scale)
        h = int(h / det_input.scale)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"ID: {CLASSES_NAMES[det.class_id]} Conf: {det.score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, f"ID: {CLASSES_NAMES[det.class_id]} Conf: {det.score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()