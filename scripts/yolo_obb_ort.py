import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
CLASSES_NAMES  = ["plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"]

@dataclass
class YOLOInput:
    image: np.ndarray
    blob: np.ndarray
    top: int
    bottom: int
    left: int
    right: int
    scale: float


@dataclass
class YOLOOBBOutput:
    box: list[list[int]] # 四个点坐标列表
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
    return YOLOInput(image=src, blob=blob, top=top, bottom=bottom, left=left, right=right, scale=scale)


def postProcess(outputs: list[np.ndarray]) -> list[YOLOOBBOutput]:
    assert len(outputs) == 1, "Expected a single output from the model."
    assert outputs[0].ndim == 3, "Expected output to be a 3D array."
    predictions = outputs[0][0]
    predictions = predictions.transpose(1, 0)

    class_ids = []
    boxes = []
    confidences = []

    for pred in predictions:
        score = pred[4:-1]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > CONF_THRESHOLD:
            cx, cy, w, h = pred[0:4]
            rad = pred[-1]
            angle = rad * 180.0 / np.pi  # 转为角度制
            boxes.append([[cx, cy], [int(w), int(h)], angle])
            class_ids.append(class_id)
            confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)
    results = []
    for i in indices:
        box = cv2.boxPoints(boxes[i]).tolist()
        score = confidences[i]
        class_id = class_ids[i]
        results.append(YOLOOBBOutput(box=box, score=score, class_id=class_id))
    # print("注意 YOLOOBBOutput 中的 box 坐标是相对于输入图片的坐标，需根据预处理时的缩放和填充进行映射回原图坐标。")
    return results

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
    model_path = "res/yolo/yolo11n-obb.onnx"
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

    # Warm up
    dummy_input = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
    session.run(None, {session.get_inputs()[0].name: dummy_input})

    # Run inference for {runs} times to measure performance
    image_path = "res/P0015.jpg"
    runs = 100
    for _ in range(runs):
        image = cv2.imread(image_path)
        input_size = (1024, 1024)
        det_input = preProcess(image, input_size[0])
        outputs = session.run([output_name], {input_name: det_input.blob})
        results = postProcess(outputs)

    for obb in results:
        box = obb.box
        score = obb.score
        class_id = obb.class_id

        map_box = []
        for i in box:
            i[0] = int((i[0] - det_input.left) / det_input.scale)
            i[1] = int((i[1] - det_input.top) / det_input.scale)
            map_box.append(i)
        cv2.polylines(image, [np.array(map_box, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
