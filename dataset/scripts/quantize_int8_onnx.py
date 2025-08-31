import os
import glob
import random
from typing import List

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image
import numpy as np

# Minimal preproc matching YOLOv8 default: resize+letterbox to 640, BGR->RGB if needed
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
CALIB_MAX = int(os.getenv("CALIB_MAX", "200"))

class YoloImageFolder(CalibrationDataReader):
    def __init__(self, folder: str, input_name: str = "images"):
        self.folder = folder
        self.input_name = input_name
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, e)))
        random.shuffle(files)
        self.files = files[:CALIB_MAX]
        self.iter = iter(self.files)

    def get_next(self):
        try:
            path = next(self.iter)
        except StopIteration:
            return None
        img = Image.open(path).convert("RGB")
        img = letterbox(img, new_shape=IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32)
        arr = arr / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, 0)  # NCHW
        return {self.input_name: arr}

    def rewind(self):
        self.iter = iter(self.files)


def letterbox(img: Image.Image, new_shape=640, color=(114, 114, 114)) -> Image.Image:
    w, h = img.size
    if isinstance(new_shape, int):
        new_w = new_h = new_shape
    else:
        new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))

    img_resized = img.resize((nw, nh))
    new_img = Image.new("RGB", (new_w, new_h), color)
    new_img.paste(img_resized, ((new_w - nw) // 2, (new_h - nh) // 2))
    return new_img


def main():
    model_in = os.getenv("MODEL_IN", r"runs/detect/train3/weights/best.onnx")
    model_out = os.getenv("MODEL_OUT", r"runs/detect/train3/weights/best-int8.onnx")
    calib_dir = os.getenv("CALIB_DIR", r"valid/images")

    onnx_model = onnx.load(model_in)
    sess_input = onnx_model.graph.input[0].name if onnx_model.graph.input else "images"

    print(f"Model: {model_in}\nInput: {sess_input}\nCalib dir: {calib_dir}\nOutput: {model_out}")

    dr = YoloImageFolder(calib_dir, input_name=sess_input)

    quantize_static(
        model_input=model_in,
        model_output=model_out,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        op_types_to_quantize=["Conv"],
    )
    print(f"Saved INT8 model to: {model_out}")


if __name__ == "__main__":
    main()
