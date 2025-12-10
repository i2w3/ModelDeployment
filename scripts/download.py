import shutil
from pathlib import Path

from ultralytics import YOLO

model_list = [
    "yolo11n.pt",
    "yolo11n-seg.pt",
    "yolo11n-pose.pt",
    "yolo11n-cls.pt",
    "yolo11n-obb.pt"
]
output_dir = Path("res/yolo")

if __name__ == "__main__":
    for model_name in model_list:
        model = YOLO(model_name)
        onnx_path = model.export(format="onnx")
        shutil.move(onnx_path, output_dir / model_name.replace(".pt", ".onnx"))
        shutil.move(model_name, output_dir)