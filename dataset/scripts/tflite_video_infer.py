import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from postprocess_yolov8 import decode_yolov8_output, nms, scale_coords

# Script: TFLite video inference (single-class YOLOv8 output format (1,5,8400) or (1,8400,5))

def letterbox(im, new_shape=640, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape / w, new_shape / h)
    nw, nh = int(round(w*r)), int(round(h*r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=im.dtype)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top+nh, left:left+nw] = im_r
    return canvas, r, left, top

def load_interpreter(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(f"[INFO] Model={model_path}\n       Input shape={inp['shape']} dtype={inp['dtype']} quant={inp.get('quantization')}\n       Output shape={out['shape']} dtype={out['dtype']} quant={out.get('quantization')}")
    return interpreter, inp, out

def prepare_input(frame_bgr, imgsz, inp_detail):
    img, r, pad_w, pad_h = letterbox(frame_bgr, imgsz)
    arr = img.astype(np.float32) / 255.0  # (H,W,3)
    in_shape = inp_detail['shape']
    in_dtype = inp_detail['dtype']
    if len(in_shape) != 4:
        raise ValueError(f"Unsupported input rank: {in_shape}")
    # Assume NHWC unless channel dim is second
    if in_shape[1] == 3 and in_shape[2] == imgsz:
        # NCHW expected
        arr = np.transpose(arr, (2,0,1))[None, ...]
    else:
        arr = arr[None, ...]
    if in_dtype in (np.uint8, np.int8):
        scale, zero = inp_detail['quantization']
        if scale == 0:
            scale = 1.0
        q = arr / scale + zero
        if in_dtype == np.uint8:
            arr = np.clip(np.round(q), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(np.round(q), -128, 127).astype(np.int8)
    return arr, r, pad_w, pad_h

def process_output(raw_out, out_detail):
    out = raw_out
    if out_detail['dtype'] in (np.uint8, np.int8):
        s, z = out_detail['quantization']
        if s == 0:
            s = 1.0
        out = (out.astype(np.float32) - z) * s
    if out.ndim == 3 and out.shape[1] == 8400 and out.shape[2] == 5:
        out = np.transpose(out, (0,2,1))
    return out

def run_video(args):
    interpreter, inp, out = load_interpreter(args.model)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source vidéo: {args.source}")
    os.makedirs(args.outdir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.source))[0] + '_pred.mp4')
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (int(cap.get(3)), int(cap.get(4))))
    frame_id = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        arr, r, pad_w, pad_h = prepare_input(frame, args.imgsz, inp)
        interpreter.set_tensor(inp['index'], arr)
        t0 = time.time()
        interpreter.invoke()
        infer_t = (time.time() - t0)*1000
        times.append(infer_t)
        raw = interpreter.get_tensor(out['index'])
        out_tensor = process_output(raw, out)
        boxes, scores = decode_yolov8_output(out_tensor, args.conf)
        if boxes.size:
            keep = nms(boxes, scores, args.iou)
            boxes = boxes[keep]
            scores = scores[keep]
            # inverse letterbox scaling
            gain = min(args.imgsz / frame.shape[1], args.imgsz / frame.shape[0])
            # Actually scale_coords needs shapes (h,w)
            boxes = scale_coords((args.imgsz, args.imgsz), boxes.copy(), (frame.shape[0], frame.shape[1]))
            for b, s in zip(boxes, scores):
                x1,y1,x2,y2 = map(int, b.tolist())
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,140,255), 2)
                cv2.putText(frame, f"obj {s:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,255), 1, cv2.LINE_AA)
        writer.write(frame)
        if args.max_frames and frame_id >= args.max_frames:
            break
        if frame_id % 50 == 0:
            print(f"[INFO] Frame {frame_id}, last infer {infer_t:.1f} ms")
    cap.release()
    writer.release()
    if times:
        print(f"[STATS] Frames={len(times)} mean={np.mean(times):.2f} ms min={np.min(times):.2f} ms max={np.max(times):.2f} ms")
    print(f"[DONE] Output vidéo: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 TFLite Video Inference")
    ap.add_argument('--model', default='runs/detect/train3/weights/yolov8n_bag_int8.tflite', help='Chemin modèle .tflite')
    ap.add_argument('--source', required=True, help='Chemin vidéo (mp4, avi, etc.)')
    ap.add_argument('--outdir', default='runs/tflite_video', help='Dossier sortie')
    ap.add_argument('--imgsz', type=int, default=640, help='Taille entrée carré')
    ap.add_argument('--conf', type=float, default=0.25, help='Seuil confiance')
    ap.add_argument('--iou', type=float, default=0.45, help='Seuil IOU NMS')
    ap.add_argument('--max-frames', type=int, default=0, help='Limiter nombre de frames (0 = toutes)')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_video(args)
