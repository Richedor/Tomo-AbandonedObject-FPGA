import sys
from pathlib import Path
import zipfile
import shutil

base = Path(__file__).resolve().parents[1]
calib_txt = base / "runs" / "detect" / "train3" / "calib.txt"
zip_path = base / "runs" / "detect" / "train3" / "calib_images_200.zip"
calib_tmp_dir = base / "runs" / "detect" / "train3" / "calib_images"

if calib_tmp_dir.exists():
    try:
        shutil.rmtree(calib_tmp_dir)
        print(f"CLEANED: {calib_tmp_dir}")
    except Exception as e:
        print(f"WARN: impossible de supprimer {calib_tmp_dir}: {e}")

if not calib_txt.exists():
    print(f"ERROR: calib.txt introuvable: {calib_txt}")
    sys.exit(1)
if not zip_path.exists():
    print(f"ERROR: zip introuvable: {zip_path}")
    sys.exit(1)

# Construire la table nom_de_fichier -> chemin complet d'origine
mapping = {}
for line in calib_txt.read_text(encoding="utf-8").splitlines():
    p = line.strip().strip('"')
    if not p:
        continue
    src = Path(p)
    mapping[src.name] = src

restored = 0
skipped_exists = 0
not_mapped = 0

with zipfile.ZipFile(zip_path, 'r') as zf:
    for name in zf.namelist():
        target = mapping.get(name)
        if target is None:
            not_mapped += 1
            continue
        if target.exists():
            skipped_exists += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src_f, open(target, 'wb') as dst_f:
            dst_f.write(src_f.read())
        restored += 1

print(f"RESTORE_DONE zip={zip_path} restored={restored} exists={skipped_exists} unmapped={not_mapped}")
