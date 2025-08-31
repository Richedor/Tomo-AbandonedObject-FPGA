from ultralytics import YOLO
import os

# Exporte le modèle YOLOv8 entraîné en TensorFlow SavedModel (sans NMS)
# Entrée/sortie par défaut: 640x640, float32 [0..1]
# Sortie: dossier SavedModel (ex: runs/detect/train3/weights/best_saved_model)

MODEL_PT = os.getenv("MODEL_PT", r"runs/detect/train3/weights/best.pt")
IMGSZ = int(os.getenv("IMGSZ", "640"))

m = YOLO(MODEL_PT)
print(f"Export TF SavedModel depuis: {MODEL_PT} (imgsz={IMGSZ})")
path = m.export(format="tf", imgsz=IMGSZ, keras=False)  # keras=False -> SavedModel
print("SavedModel:", path)
