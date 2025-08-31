import os, glob
import numpy as np
import tensorflow as tf

# Quantification INT8 post-training TFLite depuis un SavedModel TF
# - Input: SavedModel export TF (sans NMS)
# - Représentative dataset: images dans CALIB_DIR (200 par défaut)

SAVED = os.getenv("SAVED", r"runs/detect/train3/weights/best_saved_model")
OUT = os.getenv("OUT", r"runs/detect/train3/weights/yolov8n_int8.tflite")
CALIB_DIR = os.getenv("CALIB_DIR", r"valid/images")
IMGSZ = int(os.getenv("IMGSZ", "640"))
MAXN = int(os.getenv("MAXN", "200"))


def letterbox_np(img: np.ndarray, new_shape=640):
    h, w = img.shape[:2]
    r = min(new_shape / w, new_shape / h)
    nw, nh = int(round(w*r)), int(round(h*r))
    im_r = tf.image.resize(img, (nh, nw)).numpy()
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=img.dtype)
    pad_w = (new_shape - nw) // 2
    pad_h = (new_shape - nh) // 2
    canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = im_r
    return canvas


def rep_ds():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(CALIB_DIR, e)))
    files = files[:MAXN]
    for p in files:
        im = tf.io.read_file(p)
        im = tf.image.decode_image(im, channels=3)
        im = tf.cast(im, tf.float32) / 255.0
        im = letterbox_np(im.numpy(), IMGSZ).astype(np.float32)
        yield [np.expand_dims(im, 0)]


print(f"SavedModel: {SAVED}\nOut: {OUT}\nCalib: {CALIB_DIR} ({MAXN})")
conv = tf.lite.TFLiteConverter.from_saved_model(SAVED)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_ds
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Si le graphe supporte l'entier pur:
conv.inference_input_type = tf.uint8
conv.inference_output_type = tf.uint8

# Option: forcer un nombre minimal d'opérateurs quantifiés
# conv._experimental_calibrate_only = False

print("Conversion...")
tfl = conv.convert()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "wb").write(tfl)
print("OK ->", OUT)
