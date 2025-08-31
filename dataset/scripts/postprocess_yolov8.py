import math
from typing import Tuple, List
import numpy as np

# Utilitaires de post-traitement pour YOLOv8 export ONNX sans NMS
# Gère sorties (1, C, N) où C peut être 5 (xywh+conf unique) ou >=6 (xywh+obj(+classes))


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_thres]
    return keep


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    # box: (4,), boxes: (N,4) in xyxy
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def scale_coords(img_shape: Tuple[int, int], boxes: np.ndarray, orig_shape: Tuple[int, int]) -> np.ndarray:
    # img_shape: (h, w) of network input (e.g., 640x640)
    # orig_shape: (h0, w0) original image size
    gain = min(img_shape[0] / orig_shape[0], img_shape[1] / orig_shape[1])
    pad_w = (img_shape[1] - orig_shape[1] * gain) / 2
    pad_h = (img_shape[0] - orig_shape[0] * gain) / 2

    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= gain

    boxes[:, 0] = boxes[:, 0].clip(0, orig_shape[1] - 1)
    boxes[:, 1] = boxes[:, 1].clip(0, orig_shape[0] - 1)
    boxes[:, 2] = boxes[:, 2].clip(0, orig_shape[1] - 1)
    boxes[:, 3] = boxes[:, 3].clip(0, orig_shape[0] - 1)
    return boxes


def decode_yolov8_output(out: np.ndarray, conf_thres: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    # out: (1, C, N)
    if out.ndim == 3:
        out = out[0]
    C, N = out.shape
    # Cases:
    # - C == 5: [x,y,w,h,conf] single-class
    # - C >= 6: [x,y,w,h,obj,(cls...)]
    if C == 5:
        xywh = out[:4].T  # (N,4)
        conf = out[4]
        cls_id = np.zeros_like(conf, dtype=np.int32)
        conf_mask = conf >= conf_thres
        boxes = xywh2xyxy(xywh)
        return boxes[conf_mask], conf[conf_mask]
    else:
        xywh = out[:4].T
        obj = out[4]
        cls = out[5:]
        cls_scores = cls.max(axis=0)
        conf = obj * cls_scores
        conf_mask = conf >= conf_thres
        boxes = xywh2xyxy(xywh)
        return boxes[conf_mask], conf[conf_mask]
