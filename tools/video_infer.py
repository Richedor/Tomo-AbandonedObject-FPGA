import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Importer post-traitement depuis scripts racine
import sys
ROOT = Path(__file__).resolve().parents[0].parents[0]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.append(str(ROOT / 'scripts'))
from postprocess_yolov8 import decode_yolov8_output, nms, scale_coords

def letterbox(im, new_shape=640, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape / w, new_shape / h)
    nw, nh = int(round(w*r)), int(round(h*r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=im.dtype)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top+nh, left:left+nw] = im_r
    return canvas

def load_interpreter(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(f"[INFO] Model={model_path}\n       Input shape={inp['shape']} dtype={inp['dtype']} quant={inp.get('quantization')}\n       Output shape={out['shape']} dtype={out['dtype']} quant={out.get('quantization')}")
    return interpreter, inp, out

def prepare_input(frame_bgr, imgsz, inp_detail):
    img = letterbox(frame_bgr, imgsz)
    arr = img.astype(np.float32) / 255.0
    in_shape = inp_detail['shape']
    in_dtype = inp_detail['dtype']
    # NCHW vs NHWC
    if len(in_shape) == 4 and in_shape[1] == 3 and in_shape[2] == imgsz:
        arr = np.transpose(arr, (2,0,1))[None, ...]
    else:
        arr = arr[None, ...]
    if in_dtype in (np.uint8, np.int8):
        s, z = inp_detail['quantization']
        if s == 0: s = 1.0
        q = arr / s + z
        if in_dtype == np.uint8:
            arr = np.clip(np.round(q), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(np.round(q), -128, 127).astype(np.int8)
    return arr

def process_output(raw_out, out_detail):
    out = raw_out
    if out_detail['dtype'] in (np.uint8, np.int8):
        s, z = out_detail['quantization']
        if s == 0: s = 1.0
        out = (out.astype(np.float32) - z) * s
    if out.ndim == 3 and out.shape[1] == 8400 and out.shape[2] == 5:
        out = np.transpose(out, (0,2,1))
    return out

def run_video(args):
    interpreter, inp, out = load_interpreter(args.model)
    if os.path.isdir(args.source):
        videos = [str(p) for p in Path(args.source).glob('*.mp4')]
    else:
        videos = [args.source]
    os.makedirs(args.outdir, exist_ok=True)
    for vid in videos:
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print(f"[WARN] Impossible d'ouvrir: {vid}")
            continue
        base = os.path.splitext(os.path.basename(vid))[0]
        out_path = os.path.join(args.outdir, base + '_pred.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (int(cap.get(3)), int(cap.get(4))))
        frame_id = 0
        times=[]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            arr = prepare_input(frame, args.imgsz, inp)
            interpreter.set_tensor(inp['index'], arr)
            t0 = time.time()
            interpreter.invoke()
            infer_t = (time.time()-t0)*1000
            times.append(infer_t)
            raw = interpreter.get_tensor(out['index'])
            out_tensor = process_output(raw, out)
            boxes, scores = decode_yolov8_output(out_tensor, args.conf)
            if boxes.size:
                keep = nms(boxes, scores, args.iou)
                boxes = boxes[keep]
                scores = scores[keep]
                boxes = scale_coords((args.imgsz, args.imgsz), boxes.copy(), (frame.shape[0], frame.shape[1]))
                for b, s in zip(boxes, scores):
                    x1,y1,x2,y2 = map(int, b.tolist())
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,140,255), 2)
                    cv2.putText(frame, f"obj {s:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,255), 1, cv2.LINE_AA)
            writer.write(frame)
            if args.max_frames and frame_id >= args.max_frames:
                break
            if frame_id % 50 == 0:
                print(f"[INFO] {base} frame {frame_id} last {infer_t:.1f} ms")
        cap.release()
        writer.release()
        if times:
            print(f"[STATS] {base} frames={len(times)} mean={np.mean(times):.2f} ms min={np.min(times):.2f} ms max={np.max(times):.2f} ms")
        print(f"[DONE] Video sortie: {out_path}")

def parse_args():
    ap = argparse.ArgumentParser(description='Inference vidéo YOLOv8 TFLite')
    ap.add_argument('--model', default='runs/detect/train3/weights/yolov8n_bag_int8.tflite')
    ap.add_argument('--source', required=True, help='Chemin vidéo ou dossier de vidéos')
    ap.add_argument('--outdir', default='runs/tflite_video')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--max-frames', type=int, default=0)
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_video(args)
