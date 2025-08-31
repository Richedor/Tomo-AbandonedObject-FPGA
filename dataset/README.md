## Dataset Overview

Ce dossier regroupe les informations sur la constitution du dataset utilisé pour entraîner le modèle YOLOv8n.

Sources:
- Roboflow "Abandoned Object Detection" (licence CC BY 4.0)
- Éventuel sous-ensemble COCO (person / bag) pour extension multi-classes (WIP)

Scripts utiles (dans `dataset/scripts/`):
- `roboflow_download.py` : téléchargement / export
- `make_calib_list.py` : liste calibration quantization INT8
- `zip_calib_images.py` / `restore_calib_images.py`
- `tflite_infer.py` : inférence images TFLite
- `tflite_video_infer.py` : inférence vidéo TFLite

Anciennes notices:
- Voir `README.dataset.txt`
- Voir `README.roboflow.txt`

Prochaines étapes:
- Fusion multi-classes (person + object)
- Nettoyage / équilibrage classes
- Génération métriques temporelles (tracking) pour logique abandon
