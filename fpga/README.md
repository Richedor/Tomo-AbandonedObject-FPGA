## FPGA Overview

Plateforme: Zybo Z7-10 (XC7Z010)

IP prévues / réalisées:
- `ResizeLetterbox320` (HLS) : redimensionnement + letterbox
- `MotionGate` (HLS, WIP) : masque de mouvement pour ignorer frames statiques

Arborescence:
- `vivado/` : scripts tcl, contraintes XDC, captures block design
- `vitis/` : application PS (C/C++), Makefile
- `hls/` : projets HLS (sources, scripts, rapports)

Prochaines étapes:
- Intégrer pipeline MIPI → AXI Stream → HLS IP → DMA → PS
- Générer bitstream + export XSA
- Compiler application Vitis (chargement modèle quantifié)
