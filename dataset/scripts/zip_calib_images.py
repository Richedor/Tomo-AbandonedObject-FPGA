import sys
from pathlib import Path
import zipfile

base = Path(__file__).resolve().parents[1]
calib_txt = base / "runs" / "detect" / "train3" / "calib.txt"
out_zip = base / "runs" / "detect" / "train3" / "calib_images_200.zip"

copied = 0
missing = 0

if not calib_txt.exists():
    print(f"ERROR: calib.txt introuvable: {calib_txt}")
    sys.exit(1)

lines = calib_txt.read_text(encoding="utf-8").splitlines()

with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    for line in lines:
        p = line.strip().strip('"')
        if not p:
            continue
        src = Path(p)
        if src.exists():
            # Ajouter le fichier avec son nom seulement dans l'archive
            zf.write(src, arcname=src.name)
            copied += 1
        else:
            missing += 1

print(f"ZIP_OK path={out_zip} copied={copied} missing={missing}")
