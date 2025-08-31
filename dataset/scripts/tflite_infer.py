import os
import sys
import glob
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from postprocess_yolov8 import decode_yolov8_output, nms, scale_coords

MODEL_DEFAULT = r"runs/detect/train3/weights/yolov8n_bag_int8.tflite"
MODEL = os.getenv("MODEL", MODEL_DEFAULT)
# argument positionnel optionnel: chemin modèle
if len(sys.argv) > 1 and sys.argv[1].endswith('.tflite'):
    MODEL = sys.argv[1]

# auto-détection si chemin absent
if not os.path.isfile(MODEL):
    candidates = sorted(glob.glob("runs/detect/train3/weights/*.tflite"))
    if candidates:
        print(f"[INFO] Modèle spécifié introuvable, utilisation de {candidates[0]}")
        MODEL = candidates[0]
SOURCE = os.getenv("SOURCE", r"test/images")
OUTDIR = os.getenv("OUTDIR", r"runs/tflite_predict")
IMGSZ = int(os.getenv("IMGSZ", "640"))
CONF = float(os.getenv("CONF", "0.25"))
IOU = float(os.getenv("IOU", "0.45"))

os.makedirs(OUTDIR, exist_ok=True)

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Info debug
try:
    print(f"[INFO] Input shape={input_details[0]['shape']} dtype={input_details[0]['dtype']} quant={input_details[0].get('quantization')}")
    print(f"[INFO] Output shape={output_details[0]['shape']} dtype={output_details[0]['dtype']} quant={output_details[0].get('quantization')}")
except Exception:
    pass


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
    im = letterbox(im0, IMGSZ)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)

    in_shape = input_details[0]['shape']
    in_dtype = input_details[0]['dtype']

    # Déterminer format attendu: NHWC (1,H,W,3) ou NCHW (1,3,H,W)
    if len(in_shape) == 4:
        if in_shape[1] == 3 and in_shape[2] == IMGSZ:  # (1,3,H,W)
            arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
            arr = arr[None, ...]
        else:  # supposer NHWC
            arr = arr[None, ...]  # (1,H,W,3)
    else:
        raise ValueError(f"Shape entrée non supportée: {in_shape}")

    # Quantification entrée (uint8 / int8)
    if in_dtype in (np.uint8, np.int8):
        scale, zero = input_details[0]['quantization']
        if scale == 0:
            scale = 1.0
        q = arr / scale + zero
        if in_dtype == np.uint8:
            q = np.clip(np.round(q), 0, 255).astype(np.uint8)
        else:
            q = np.clip(np.round(q), -128, 127).astype(np.int8)
        arr = q

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])

    # Adapter éventuellement format (1,8400,5) -> (1,5,8400)
    if out.ndim == 3 and out.shape[1] == 8400 and out.shape[2] == 5:
        out = np.transpose(out, (0, 2, 1))

    # Déquantifier sortie éventuelle
    out_dtype = output_details[0]['dtype']
    if out_dtype in (np.uint8, np.int8):
        s, z = output_details[0]['quantization']
        if s == 0:
            s = 1.0
        out = (out.astype(np.float32) - z) * s

    boxes, scores = decode_yolov8_output(out, CONF)
    if boxes.size:
        keep = nms(boxes, scores, IOU)
        boxes = boxes[keep]
        scores = scores[keep]
        boxes = scale_coords((IMGSZ, IMGSZ), boxes.copy(), (h0, w0))

    draw = ImageDraw.Draw(im0)
    for b, s in zip(boxes, scores):
        x1,y1,x2,y2 = b.tolist()
        draw.rectangle([x1,y1,x2,y2], outline="orange", width=2)
        draw.text((x1, y1-10), f"obj {s:.2f}", fill="orange")

    out_path = os.path.join(OUTDIR, os.path.basename(path))
    im0.save(out_path)
    print("Saved:", out_path)


def main():
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = [os.path.join(SOURCE, p) for p in os.listdir(SOURCE) if os.path.splitext(p)[1].lower() in exts]
    for p in paths[:20]:
        run_image(p)


if __name__ == "__main__":
    main()
