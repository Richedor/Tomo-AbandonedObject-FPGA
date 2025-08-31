## Training Notes

Modèle base: YOLOv8n (Ultralytics) single-class (objet / bag).

Exports:
- ONNX (avec / sans NMS)
- TFLite INT8 (post-training quantization)

À ajouter:
- Notebooks `notebooks/` (entraînement + export quantization)
- Multi-class (person + bag) pour logique abandon robuste
- Pruning / distillation éventuelle

Commandes (exemple):
```
yolo train model=yolov8n.pt data=training/data.yaml epochs=... imgsz=640
```

Quantization future: calibration set + export INT8 optimisé.
