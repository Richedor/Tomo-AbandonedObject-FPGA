import os, glob, random

CALIB_DIR = os.getenv("CALIB_DIR", r"valid/images")
OUT = os.getenv("OUT", r"runs/detect/train3/calib.txt")
MAXN = int(os.getenv("MAXN", "200"))

exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
files = []
for e in exts:
    files.extend(glob.glob(os.path.join(CALIB_DIR, e)))
random.shuffle(files)
files = files[:MAXN]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for p in files:
        f.write(os.path.abspath(p) + "\n")
print(f"Wrote {len(files)} lines to {OUT}")
