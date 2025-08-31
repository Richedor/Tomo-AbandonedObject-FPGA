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

---

## 🏗 Architecture & Roadmap

### Vision & Contraintes
- Cas d’usage : détection de **colis abandonné** (gare, hall) avec caméra fixe.
- Contraintes : ≥10–15 FPS perçus (avec frame gating), basse conso, coût modéré, reproductible, open-source.
- Plateforme : **Zybo Z7-10 (Zynq-7010)** + **Pcam 5C (MIPI)**.

### Vue d’Ensemble
```
[Caméra Pcam 5C] 
	 -> [PL/FPGA: capture MIPI -> (WIP) MotionGate -> Resize+Letterbox]
	 -> [PS/ARM A9: YOLOv8n INT8 inference + NMS + logique "abandon"]
	 -> [Sorties: alerte, overlay GUI, logs métriques]
```

### Rôle du PL (FPGA)
| Fonction | Détail | Justification |
|----------|--------|---------------|
| Capture MIPI | Interface Pcam, format AXI Stream | Indispensable dans le PL |
| Resize + Letterbox (HLS) | Passage natif → 320×320 (ou 256) | Déleste CPU, pipeline constant |
| MotionGate (WIP) | Ouvre une “fenêtre” d’inférence quand mouvement | Réduit inférences inutiles |

### Rôle du PS (ARM Cortex-A9)
| Composant | Rôle |
|-----------|------|
| TFLite / ONNX Runtime INT8 | Inférence YOLOv8n optimisée |
| NMS + logique abandon | Détection immobilité + absence de personne |
| GUI / Logs | Démo, métriques, alertes |

### Découpage PL/PS – Principes
1. Opérations régulières, streamables → **PL** (pipeline, latence stable).
2. Contrôle complexe, logique métier, ML dynamique → **PS** (flexibilité).

### Choix Techniques (Résumé)
- **YOLOv8n + INT8** : équilibre précision / latence / mémoire pour ARM A9.
- **320×320** (cible) : compromis petits objets vs coût calcul.
- **Motion gating** : économise énergie → inférence uniquement sur frames “actives”.
- **Prétraitements en PL** : resize/letterbox = goulot CPU typique.
- **GUI PC** avant embarqué : itérer sur logique “abandon” (seuils temps + distance).

### Données & Modèle
- Dataset fusion : Roboflow Abandoned Objects + sous-ensemble COCO (classes : person, backpack, briefcase, handbag, suitcase).
- Entraînement : Ultralytics YOLOv8 (Colab) → export ONNX + TFLite INT8.
- Prochaine étape : ajout tracking (SORT / ByteTrack) + calibrage logique d’abandon.

### Chronologie (Étapes)
1. Définition périmètre & architecture PL/PS.
2. Fusion dataset + splits propres.
3. Entraînement YOLOv8n + exports ONNX/TFLite INT8.
4. Démo PC (GUI + logique abandon initiale).
5. IP HLS ResizeLetterbox320 (OK) / MotionGate (WIP).
6. Intégration Vivado (capture + pipeline) + App PS (chargement modèle).
7. À venir : finalisation MotionGate, tracking, tuning FP/FN, tests terrain, packaging open-source.

### Reproductibilité (Open-Source)
| Élément | Fourniture |
|---------|------------|
| Matériel | Zybo Z7-10, Pcam 5C, µSD |
| HLS | `fpga/hls/ResizeLetterbox320/` + (WIP) `MotionGate/` |
| Vivado | Scripts `.tcl`, contraintes `.xdc`, captures block design |
| Vitis / PS | App C/C++, Makefile, chargement TFLite |
| Modèle | Poids quantifiés (≤95 MB) + notebooks entraînement |
| Scripts | `tools/pack_tomo_repo.py`, futurs `build_pl.tcl`, `setup_ps.sh` |
| Docs | Diagrammes (`docs/`), rapport (`report/`) |

### Performance Cible
| Scénario | Objectif |
|----------|----------|
| Sans gating | ~3–8 FPS (brut) |
| Avec gating | 10–15 FPS perçus (inférence partielle) |
| Latence alerte | < 1–2 s |

### Risques & Mitigations
| Risque | Mitigation |
|--------|------------|
| Petits objets / occlusion | Taille 320 + data aug + tracking |
| Variabilité lumière | Seuils MotionGate adaptatifs |
| Charge CPU excessive | Frame skip + gating + éventuel 256×256 |
| Faux positifs “abandon” | Distance personne + temps d’observation |

### Roadmap (Checklist)
- [x] YOLOv8n entraîné & export INT8
- [x] Prétraitement HLS (ResizeLetterbox320)
- [ ] MotionGate HLS finalisé
- [ ] Tracking (SORT / ByteTrack) intégré
- [ ] Modèle multi-classes (person + bag types)
- [ ] Intégration complète Vivado/Vitis
- [ ] Tests terrain & tuning
- [ ] Rapport final + vidéo démo

---

## 🔁 How to Reproduce (Draft)
1. Cloner :
	```
	git clone https://github.com/<user>/Tomo-AbandonedObject-FPGA.git
	cd Tomo-AbandonedObject-FPGA
	```
2. (Optionnel) Récupérer poids (si non inclus) depuis Releases.
3. GUI PC :
	```
	cd gui
	pip install -r requirements.txt
	python test_gui.py --video ../samples/sequence.mp4
	```
4. Entraînement / Export (voir `training/README.md` + notebooks).
5. Générer IP HLS : ouvrir `fpga/hls/ResizeLetterbox320` (Vitis HLS), exporter IP.
6. Vivado : script `fpga/vivado/build_project.tcl` (à venir) → bitstream + export XSA.
7. Vitis : créer plateforme XSA, compiler app (TFLite + logique abandon).
8. Déploiement : copier BOOT + bitstream sur µSD, lancer app PS.
9. Profilage : comparer FPS avec/sans MotionGate.
10. Tests : scènes réelles diverses, relever FP/FN.

### Prochaines Améliorations
- Intégrer ByteTrack pour stabilité inter-frames.
- Ajouter heuristique “zone interdite” (ROI mask).
- Support RTSP / IP camera.
- Script unique end-to-end (build + démo).

