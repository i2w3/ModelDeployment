from pathlib import Path
import onnxruntime as ort

series_list = ["yolo11", "yolov8"]
scales_list = ["n", "s"]

model_list = [
    "{}{}.onnx",
    "{}{}-seg.onnx",
    "{}{}-pose.onnx",
    "{}{}-cls.onnx",
    "{}{}-obb.onnx"
]

model_path  = Path("res/yolo")
enable_FP32 = False

if __name__ == "__main__":
    for series in series_list:
        for scale in scales_list:
            model_names = [model.format(series, scale) for model in model_list]
            for model_name in model_names:
                model_name = Path(model_name)
                so = ort.SessionOptions()
                providers = [
                    ('TensorrtExecutionProvider', {
                        'device_id': 0,
                        'trt_max_workspace_size': 4 * 1024 * 1024 * 1024, # 4 GB
                        'trt_fp16_enable': not enable_FP32,
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': './res/yolo/trt_cache',
                        'trt_engine_cache_prefix': f'{model_name.stem}_{"fp32" if enable_FP32 else "fp16"}',
                        'trt_timing_cache_enable': True, # timing cache 加速在其它设备上建立 engine
                        'trt_timing_cache_path': './res/yolo/trt_cache/time_cache',
                        'trt_force_timing_cache': False, # 仅在与生成 timing cache 的 GPU 型号完全相同的 GPU 上使用
                    })
                ]
                session = ort.InferenceSession(str(model_path / model_name), sess_options=so, providers=providers)
                del session, so, providers