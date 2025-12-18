import shutil
from pathlib import Path

from ultralytics import YOLO

series_list = ["yolo11", "yolov8"]
scales_list = ["n", "s"]

model_list = [
    "{}{}.pt",
    "{}{}-seg.pt",
    "{}{}-pose.pt",
    "{}{}-cls.pt",
    "{}{}-obb.pt"
]
output_dir = Path("res/yolo")

if __name__ == "__main__":
    for series in series_list:
        for scale in scales_list:
            model_names = [model.format(series, scale) for model in model_list]
            for model_name in model_names:
                model = YOLO(model_name)
                onnx_path = model.export(format="onnx")
                shutil.move(onnx_path, output_dir / model_name.replace(".pt", ".onnx"))
                shutil.move(model_name, output_dir)