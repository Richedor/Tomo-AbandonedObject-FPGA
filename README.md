# Tomo — Abandoned Object Detection on FPGA (AOHW25_193)

**Competition:** AMD Open Hardware 2025 — Student / Adaptive Computing  
**Author:** Miguel Laleye  
**Supervisor:** Madani Mahdi, PhD  

## 🎯 Objective
Tomo est un système embarqué de détection de **colis abandonnés** (gares, aéroports, lieux publics).  
Il combine des **pré-traitements FPGA** (motion gating + resize/letterbox) sur **Zybo Z7-10 + Pcam 5C** et une **inférence YOLOv8n quantifiée INT8** sur l’ARM Cortex-A9.  


## 🧱 Global Architecture
```
[Caméra Pcam 5C]
-> [PL/FPGA: capture MIPI -> MotionGate (WIP) -> Resize+Letterbox]
-> [PS/ARM A9: YOLOv8n INT8 inference + NMS + logique "abandon"]
-> [Sorties: alerte, overlay GUI, logs métriques]
```

### Rôle du PL (FPGA)
| Fonction | Détail | Justification |
|----------|--------|---------------|
| Capture MIPI | Interface Pcam vers AXI Stream | Indispensable côté PL |
| Resize + Letterbox (HLS) | Mise au format 320×320 | Pipeline constant, décharge CPU |
| MotionGate (WIP) | Déclenche inférence si mouvement | Réduction charge CPU/énergie |

### Rôle du PS (ARM Cortex-A9)
| Composant | Rôle |
|-----------|------|
| Runtime TFLite / ONNX (INT8) | Inférence YOLOv8n |
| NMS + logique "abandon" | Temps d’immobilité + absence de personne |
| GUI / Logs | Démo PC, alertes, métriques |


## ✅ Current Status
- [x] Dataset fusionné (Roboflow Abandoned Objects + COCO subset)  
- [x] Entraînement YOLOv8n + export ONNX/TFLite INT8  
- [x] GUI PC (Tkinter/OpenCV) avec logique “colis abandonné”  
- [x] IP HLS **ResizeLetterbox320** générée  
- [ ] IP HLS **MotionGate** en développement  
- [ ] Intégration Zybo Z7-10 + Pcam (Vivado/Vitis)  
- [ ] Tests terrain et tuning FP/FN  

---

## 📂 Repository Structure
```
.
├─ dataset/            # scripts, notes dataset (Roboflow + COCO)
├─ training/           # config YOLOv8, notebooks entraînement/export
├─ gui/                # test_gui.py + requirements.txt (démo PC)
├─ fpga/
│  ├─ vivado/          # .xdc, .tcl, captures block design
│  ├─ vitis/           # app PS (C/C++), Makefile, notes
│  └─ hls/             # ResizeLetterbox320, MotionGate (WIP)
├─ docs/               # schémas, diagrammes
└─ report/             # Tomo_Report_OpenHW2025.pdf
```

---

## 🚀 Quick Start (PC demo)
```bash
cd gui
pip install -r requirements.txt
python test_gui.py --video path/to/sample.mp4
```

*La GUI affiche les détections et déclenche une alerte si un bagage reste immobile sans personne à proximité au-delà d’un seuil.*

---

## 🔧 Build (FPGA — Work In Progress)

* **Vivado 2025.1** : intégrer IP HLS *ResizeLetterbox320* + *MotionGate* (WIP), lier la Pcam (MIPI), générer le bitstream.
* **Vitis Unified IDE 2025.1** : app PS (chargement modèle INT8, NMS, logique métier).

> Notes détaillées dans `fpga/vivado/project_notes.md` et `fpga/hls/*/ip_packaging_report.md`.

---

## 🧪 Training

* YOLOv8n (Ultralytics), quantification INT8, exports ONNX et TFLite.
* Notebooks d’entraînement sous `training/notebooks/`.
* Poids > 95 MB non inclus dans Git (Git LFS recommandé).

---

## 🏗 Architecture & Roadmap

### Vision & Contraintes

* **Cas d’usage** : détection colis abandonné avec caméra fixe.
* **Contraintes** : ≥10–15 FPS perçus (grâce au motion gating), basse conso, coût modéré, open-source, facile à reproduire.
* **Plateforme** : Zybo Z7-10 (Zynq-7010) + Pcam 5C.

### Découpage PL/PS

* Opérations régulières, streamables → **PL** (resize, gating).
* Contrôle complexe, logique métier, inférence ML → **PS**.

### Choix Techniques

* **YOLOv8n + INT8** : petit modèle adapté à l’ARM A9.
* **Entrée 320×320** : équilibre petits objets vs latence.
* **Motion gating** : évite des inférences inutiles → gain conso.
* **Prétraitements en PL** : délestage CPU.
* **GUI PC** : itération rapide sur logique d’abandon.

### Données & Modèle

* Dataset : Roboflow Abandoned Objects + COCO subset (classes : *person, backpack, briefcase, handbag, suitcase*).
* Entraînement : Ultralytics YOLOv8 sur Colab → export INT8.
* Prochaine étape : ajouter tracking (SORT/ByteTrack) et calibrage de la logique abandon.

### Chronologie

1. Définition périmètre & archi PL/PS.
2. Fusion dataset.
3. Entraînement YOLOv8n + exports.
4. Démo PC (GUI + logique abandon).
5. IP ResizeLetterbox320 (OK).
6. IP MotionGate (en cours).
7. Intégration Vivado/Vitis.
8. Tests terrain, tuning FP/FN.

### Reproductibilité

* Matériel : Zybo Z7-10, Pcam 5C, µSD.
* HLS : sources & TCL fournis.
* Vivado : scripts `.tcl` + `.xdc`.
* Vitis : app C/C++ + Makefile.
* Modèle : poids INT8 (≤95 MB) + notebooks.
* Docs : schémas `docs/`, rapport `report/`.

### Performances attendues

| Scénario       | Objectif         |
| -------------- | ---------------- |
| Sans gating    | ~3–8 FPS (brut)  |
| Avec gating    | 10–15 FPS perçus |
| Latence alerte | < 1–2 s          |

---

## 🔗 Links 
* 📝 HotCRP : https://openhw2025.hotcrp.com
* 🆔 Team : **AOHW25_193**

---

## � License

MIT © 2025 Miguel Laleye


