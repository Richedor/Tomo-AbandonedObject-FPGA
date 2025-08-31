import os
import time
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw

from postprocess_yolov8 import decode_yolov8_output, nms, scale_coords

IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
CONF = float(os.getenv("CONF", "0.25"))
IOU = float(os.getenv("IOU", "0.45"))
MODEL = os.getenv("MODEL", r"runs/detect/train3/weights/best_no_nms_opset13.onnx")
SOURCE = os.getenv("SOURCE", r"test/images")
OUTDIR = os.getenv("OUTDIR", r"runs/onnx_no_nms_predict")

os.makedirs(OUTDIR, exist_ok=True)

sess = ort.InferenceSession(MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])  # fallback CPU
inp_name = sess.get_inputs()[0].name


def letterbox(im, new_shape=640, color=(114,114,114)):
    w, h = im.size
    r = min(new_shape / w, new_shape / h)
    nw, nh = int(round(w*r)), int(round(h*r))
    im_r = im.resize((nw, nh))
    canvas = Image.new("RGB", (new_shape, new_shape), color)
    canvas.paste(im_r, ((new_shape-nw)//2, (new_shape-nh)//2))
    return canvas


def run_image(path: str):
    im0 = Image.open(path).convert("RGB")
    h0, w0 = im0.height, im0.width
    im = letterbox(im0, IMG_SIZE)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))[None, ...]

    t0 = time.time()
    out = sess.run(None, {inp_name: arr})[0]
    boxes, scores = decode_yolov8_output(out, CONF)
    if boxes.size:
        keep = nms(boxes, scores, IOU)
        boxes = boxes[keep]
        scores = scores[keep]
        boxes = scale_coords((IMG_SIZE, IMG_SIZE), boxes.copy(), (h0, w0))
    dt = (time.time() - t0) * 1000

    draw = ImageDraw.Draw(im0)
    for b, s in zip(boxes, scores):
        x1,y1,x2,y2 = b.tolist()
        draw.rectangle([x1,y1,x2,y2], outline="lime", width=2)
        draw.text((x1, y1-10), f"obj {s:.2f}", fill="lime")

    out_path = os.path.join(OUTDIR, os.path.basename(path))
    im0.save(out_path)
    print(f"{path} -> {out_path} ({dt:.1f} ms)")


def main():
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = [os.path.join(SOURCE, p) for p in os.listdir(SOURCE) if os.path.splitext(p)[1].lower() in exts]
    for p in paths[:20]:
        run_image(p)


if __name__ == "__main__":
    main()
