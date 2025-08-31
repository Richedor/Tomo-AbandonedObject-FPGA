# Tomo — Abandoned Object Detection on FPGA (AOHW25_193)

**Competition:** AMD Open Hardware 2025 — Student / Adaptive Computing  
**Author:** Miguel Laleye  
**Supervisor:** Madani Mahdi, PhD

## 🎯 Objective
Tomo est un système embarqué de détection de **colis abandonnés** (gares, aéroports, lieux publics).  
Il combine **pré-traitements FPGA (motion gating + resize/letterbox)** sur **Zybo Z7-10 + Pcam 5C** et **inférence YOLOv8n quantifiée INT8** sur l’ARM Cortex-A9.

## 🧱 Architecture
- **Caméra**: Pcam 5C → **PL (FPGA)**: capture + *ResizeLetterbox320* (HLS) + *MotionGate* (HLS, WIP)  
- **PS (ARM A9)**: inférence YOLOv8n INT8 + NMS + logique “abandon”  
- **Sorties**: alerte (GUI PC pour démo), métriques et logs

```
[Camera Pcam] -> [FPGA: MotionGate + Resize] -> [PS: YOLOv8n INT8 + NMS + Logic] -> [Alert/UI]
```

## ✅ Current Status
- [x] Dataset fusionné (Roboflow Abandoned Objects + COCO subset)  
- [x] Entraînement YOLOv8n + export ONNX/TFLite INT8  
- [x] GUI PC (Tkinter/OpenCV) avec logique “colis abandonné”  
- [x] IP HLS **ResizeLetterbox320** générée  
- [ ] IP HLS **MotionGate** (en cours)  
- [ ] Intégration complète Zybo Z7-10 + Pcam 5C (Vivado/Vitis)  
- [ ] Tests terrain et tuning FP/FN

## 📂 Repository Structure
```
.
├─ dataset/            # scripts, notes dataset (Roboflow + COCO)
├─ training/           # config YOLOv8, notebooks d'entraînement/export
├─ gui/                # test_gui.py + requirements.txt (démo PC)
├─ fpga/
│  ├─ vivado/          # .xdc, .tcl, captures block design
│  ├─ vitis/           # app PS (C/C++), Makefile, notes
│  └─ hls/             # ResizeLetterbox320, MotionGate (WIP)
├─ docs/               # schémas, diagrammes, images
└─ report/             # Tomo_Report_OpenHW2025.pdf
```

## 🚀 Quick Start (PC demo)
```bash
cd gui
pip install -r requirements.txt
python test_gui.py --video path/to/sample.mp4
```
* La GUI affiche les détections et déclenche une alerte si un objet (sac/valise) reste immobile sans personne à proximité au-delà d’un seuil.

## 🔧 Build (FPGA — Work In Progress)
* **Vivado 2025.1**: intégrer IP HLS *ResizeLetterbox320* + *MotionGate* (WIP), lier Pcam (MIPI), générer bitstream.
* **Vitis Unified IDE 2025.1**: app PS (chargement modèle INT8, NMS, logique).

> Notes détaillées dans `fpga/vivado/project_notes.md` et `fpga/hls/*/ip_packaging_report.md`.

## 🧪 Training
* YOLOv8n (Ultralytics), quantification INT8, exports **ONNX/TFLite**.
* Notebooks sous `training/notebooks/`.
* Fichiers lourds (> ~95 MB) non versionnés; utiliser Git LFS si nécessaire.

## 🔗 Links (à compléter)
* 🎥 Video (≤ 2 min): https://youtu.be/XXXXX
* 📝 HotCRP submission: https://openhw2025.hotcrp.com
* 🆔 Team: **AOHW25_193**

## 📄 License
MIT © 2025 Miguel Laleye

