#!/usr/bin/env python3
"""
pack_tomo_repo.py

Prépare un dépôt propre "Tomo-AbandonedObject-FPGA" à partir du workspace courant.
Par défaut: --dry-run (aucune écriture). Utiliser --run pour exécuter réellement.
100% stdlib. Compatible Windows / macOS / Linux.

Fonctions clés:
 - Filtrage extensions pertinentes (code, docs, modèles) et seuil taille (par défaut 95 MB)
 - Exclusion dossiers bruyants (cache, build, .git, etc.) et vidéos
 - Mapping heuristique des fichiers vers l'arborescence cible
 - Génération README.md, LICENSE (MIT), .gitignore, tools/MANIFEST_SKIPPED.txt
 - Initialisation git + commit initial (si --run) + rappel GitHub

Adapter les heuristiques de mapping via la fonction map_destination().
"""
from __future__ import annotations
import argparse
import os
import sys
import shutil
import stat
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

REPO_DIR_NAME = "Tomo-AbandonedObject-FPGA"
DEFAULT_MAX_MB = 95
SKIPPED_MANIFEST_REL = Path("tools") / "MANIFEST_SKIPPED.txt"
DATASET_DIRS = {"train", "valid", "test"}

# Extensions pertinentes (en minuscules)
EXT_ALLOWED = {
    # Code / config
    ".py", ".ipynb", ".md", ".txt", ".yml", ".yaml", ".json", ".cfg", ".ini", ".tcl", ".xdc", ".v", ".sv", ".vh", ".vhf",
    ".c", ".cpp", ".h", ".hpp", ".s", ".S", ".ld", ".sh", ".ps1", ".bat",
    # Docs / images légères
    ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".dot", ".uml",
    # Modèles
    ".pt", ".pth", ".onnx", ".tflite", ".engine", ".npy", ".npz",
}
# Vidéos (toujours skip)
EXT_VIDEOS = {".mp4", ".avi", ".mov"}

# Dossiers / segments à exclure totalement (sans traverser)
EXCLUDE_DIR_SEGMENTS = {
    "__pycache__", ".venv", "venv", ".git", ".mypy_cache", ".ipynb_checkpoints", "node_modules", "dist", "build", "output", "out", "logs", "tmp", ".cache", ".vscode"
}

# Vivado / Vitis / HLS outputs patterns dir suffixes
EDA_DIR_SUFFIXES = (".runs", ".cache", ".hw", ".ip_user_files", ".sim", ".str", ".Xil")
EDA_FILE_SKIP_EXT_ALWAYS = {".jou", ".log", ".xpr", ".xsa", ".ltx", ".dcp"}
# .rpt: copier seulement si <= 5 MB sinon skip
RPT_MAX_BYTES = 5 * 1024 * 1024

# Poids / modèles: déjà gérés via EXT_ALLOWED mais respecter taille seuil sauf --include-large

LICENSE_TEXT = """MIT License\n\nCopyright (c) 2025 Miguel Laleye\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n""".strip() + "\n"

README_MD_TEMPLATE = """# Tomo — Abandoned Object Detection on FPGA (AOHW25_193)

## Contexte
Ce projet vise la détection de colis abandonnés en environnement embarqué sur carte Zybo Z7-10 (Zynq-7000) avec caméra MIPI (Pcam 5C). Point de départ : un modèle YOLOv8 (détection de colis) et extension future vers une logique spatio‑temporelle (tracking + association personnes + seuils d'abandon).

## Objectifs
1. Détection temps réel (≥ 10 FPS cible) sur plateforme hybride ARM + FPGA.
2. Réduction des ressources (quantization INT8, éventuelle pruned / tiny backbone).
3. Ajout logique d'abandon (stationnarité + absence d'interaction humaine).
4. Intégration pipeline complet : capture → preprocessing → inference → tracking → décision → GUI.

## Architecture (Vue Haut Niveau)
```
+---------+   +----------+   +-----------+   +-----------+   +----------------+
|  Pcam   |-->|  DMA /   |-->|  Preproc  |-->|  YOLO Core|-->| Tracking+Logic |
|  5C     |   |  CSI IP  |   | (resize)  |   | (FPGA or  |   |  (PS ARM)      |
+---------+   +----------+   +-----------+   |  ARM NEON)|   +--------+-------+
                                                     |                     
                                                     v                     
                                                +----------+               
                                                |  GUI /   |<--------------
                                                |  Alerts  |               
                                                +----------+               
```

## Répartition PL / PS
- PL (FPGA): Accélération potentielle (sous-ensemble convolution / backbone). Option future si portage via Vitis AI / HLS.
- PS (ARM Cortex-A9): Orchestration, post-traitement, tracking (SORT / ByteTrack), logique d'abandon, interface utilisateur.

## Matériel
- Carte: Zybo Z7-10 (XC7Z010)
- Caméra: Digilent Pcam 5C (MIPI CSI-2)
- Accélération potentielle: IP vidéo + pipeline FPGA partiel.

## État d'avancement
- [x] Détection colis (modèle YOLOv8 entraîné)
- [x] Export modèles (ONNX / TFLite INT8)
- [ ] Ajout classe personne / multi-classes
- [ ] Tracking (SORT / ByteTrack)
- [ ] Logique abandon (timer stationnarité + distance personne)
- [ ] Portage FPGA / HLS / Vitis AI
- [ ] GUI interactive (contrôles, logs, alertes)
- [ ] Rapport final & démonstration vidéo

## Lancer rapidement la GUI (placeholder)
```
python -m pip install -r gui/requirements.txt
python gui/app.py --model training/best.tflite
# Interface affiche flux caméra + statut colis (Abandonné / Surveillé)
```

## Structure Résumée
Voir l'arborescence dans ce dépôt; les gros artefacts et vidéos sont exclus (voir tools/MANIFEST_SKIPPED.txt).

## Liens
- Démo vidéo: (à venir)
- Article / Rapport: (à venir)
- HotCRP / Soumission: (placeholder)

## Licence
Projet sous licence MIT (voir `LICENSE`).

## Contribution
Issues & PR bienvenus une fois le dépôt public. Merci d'indiquer plateforme, fréquence atteinte et modifications majeures.

## Avertissement
Ce dépôt est un extrait curé. Les gros poids / vidéos ne sont pas inclus. Utiliser MANIFEST pour reconstituer si besoin (Git LFS recommandé pour futurs ajouts > 95 MB).
""".strip() + "\n"

GITIGNORE_CONTENT = """# Python
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/

# VS Code
.vscode/

# Vivado/Vitis/HLS (fichiers de build)
*.runs/
*.cache/
*.hw/
*.ip_user_files/
*.sim/
.Xil/
*.jou
*.log
*.str/
*.xpr
*.xsa
*.ltx
*.dcp
.sdk/

# ML/poids (laisser suivis si ≤ 95 MB)
*.engine

# Media lourds
*.mp4
*.avi
*.mov
""".strip() + "\n"

class FileDecision:
    def __init__(self, src: Path, dest_rel: Optional[Path], reason: str, size: int, will_copy: bool):
        self.src = src
        self.dest_rel = dest_rel  # relative to repo root
        self.reason = reason
        self.size = size
        self.will_copy = will_copy

    def human_size(self) -> str:
        return human_readable_size(self.size)


def human_readable_size(num: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num < 1024 or unit == "TB":
            return f"{num:.2f} {unit}" if unit != "B" else f"{num} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"


def is_excluded_dir(path: Path) -> bool:
    parts = {p for p in path.parts if p not in (".", "")}
    if parts & EXCLUDE_DIR_SEGMENTS:
        return True
    for p in parts:
        for suf in EDA_DIR_SUFFIXES:
            if p.endswith(suf):
                return True
    return False


def classify_extension(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in EXT_VIDEOS:
        return "video"
    if ext not in EXT_ALLOWED:
        return "unsupported"
    if ext in {".pt", ".pth", ".onnx", ".tflite", ".engine", ".npy", ".npz"}:
        return "model"
    if ext in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".dot", ".uml"}:
        return "doc-image"
    return "code"


def map_destination(rel_path: Path, classification: str) -> Path:
    name = rel_path.name.lower()
    ext = rel_path.suffix.lower()
    # Notebooks
    if ext == ".ipynb":
        return Path("training/notebooks") / rel_path.name
    # YOLO / config yaml
    if ext in {".yaml", ".yml"} and ("yolo" in name or name in {"data.yaml", "dataset.yaml", "yolov8_config.yaml"}):
        return Path("training") / rel_path.name
    # Dataset scripts heuristique
    if rel_path.parts and rel_path.parts[0].lower() == "scripts":
        if ext in {".py", ".sh", ".bat", ".ps1"}:
            return Path("dataset/scripts") / rel_path.name
    if any(tok in name for tok in ("calib", "dataset")) and ext in {".py", ".sh"}:
        return Path("dataset/scripts") / rel_path.name
    # Models
    if classification == "model":
        return Path("training") / rel_path.name
    # Vivado
    if ext in {".xdc", ".tcl"} and ("vivado" in rel_path.as_posix() or "constraints" in rel_path.as_posix()):
        return Path("fpga/vivado") / rel_path.name
    # HLS
    if (ext in {".cpp", ".hpp", ".tcl"}) and ("hls" in rel_path.as_posix() or "_hls" in rel_path.as_posix()):
        return Path("fpga/hls") / rel_path.name
    # Vitis
    if ext in {".c", ".cpp", ".h", ".hpp", ".s", ".S", ".ld"} and ("vitis" in rel_path.as_posix() or "src" in rel_path.parts):
        return Path("fpga/vitis") / rel_path.name
    if rel_path.name == "Makefile":
        return Path("fpga/vitis") / rel_path.name
    # Docs / report
    if classification == "doc-image":
        if "report" in rel_path.as_posix():
            return Path("report") / rel_path.name
        return Path("docs") / rel_path.name
    if ext == ".pdf":
        if "report" in rel_path.as_posix():
            return Path("report") / rel_path.name
        return Path("docs") / rel_path.name
    # README / root docs
    if name.startswith("readme") and ext in {".md", ".txt"}:
        return Path(rel_path.name)  # top-level
    # GUI heuristique
    if ext == ".py" and any(k in name for k in ("gui", "app", "interface")):
        return Path("gui") / rel_path.name
    # Reports (.md) maybe docs
    if ext == ".md":
        return Path("docs") / rel_path.name
    # Fallback: tools
    return Path("tools") / rel_path.name


def scan_workspace(root: Path, args) -> List[FileDecision]:
    decisions: List[FileDecision] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dir_rel = Path(dirpath).relative_to(root)
        # Exclure dossier de sortie lui-même pour éviter récursion
        if dir_rel.parts and dir_rel.parts[0] == REPO_DIR_NAME:
            continue
        # Filtrer dirnames in-place pour ne pas descendre dedans
        pruned = []
        for d in list(dirnames):
            full = Path(dirpath) / d
            if is_excluded_dir(full):
                pruned.append(d)
        for d in pruned:
            dirnames.remove(d)
        for fname in filenames:
            src = Path(dirpath) / fname
            rel_path = src.relative_to(root)
            size = src.stat().st_size
            ext = src.suffix.lower()
            # Exclure dataset images (train/ valid/ test/) sauf si --include-dataset
            if not getattr(args, 'include_dataset', False) and rel_path.parts and rel_path.parts[0] in DATASET_DIRS:
                decisions.append(FileDecision(src, None, "skip: dataset excluded", size, False))
                continue
            # Skip if inside excluded eda directories by suffix (already prevented) - safe
            # Classification
            classification = classify_extension(src)
            reason = ""
            will_copy = True
            dest_rel: Optional[Path] = None
            # Hard skip: videos
            if ext in EXT_VIDEOS:
                reason = "skip: video"
                will_copy = False
            elif classification == "unsupported":
                reason = "skip: unsupported extension"
                will_copy = False
            elif ext in EDA_FILE_SKIP_EXT_ALWAYS:
                reason = "skip: EDA build artifact"
                will_copy = False
            elif ext == ".rpt" and size > RPT_MAX_BYTES:
                reason = "skip: rpt > 5MB"
                will_copy = False
            elif size > args.max_file_mb * 1024 * 1024 and not args.include_large:
                reason = f"skip: > {args.max_file_mb}MB"
                will_copy = False
            else:
                dest_rel = map_destination(rel_path, classification)
                reason = "copy"
                if size > args.max_file_mb * 1024 * 1024 and args.include_large:
                    reason = "copy: include_large override"
            decisions.append(FileDecision(src, dest_rel, reason, size, will_copy))
    return decisions


def build_tree(decisions: List[FileDecision]) -> Dict[Path, List[FileDecision]]:
    tree: Dict[Path, List[FileDecision]] = {}
    for d in decisions:
        if not d.will_copy or not d.dest_rel:
            continue
        parent = d.dest_rel.parent
        tree.setdefault(parent, []).append(d)
    return tree


def print_preview(decisions: List[FileDecision]):
    total_copy = sum(d.size for d in decisions if d.will_copy)
    total_skip = sum(d.size for d in decisions if not d.will_copy)
    copy_count = sum(1 for d in decisions if d.will_copy)
    skip_count = len(decisions) - copy_count
    print("\n=== RÉSUMÉ PRÉLIMINAIRE ===")
    print(f"Fichiers à copier : {copy_count} ({human_readable_size(total_copy)})")
    print(f"Fichiers ignorés   : {skip_count} ({human_readable_size(total_skip)})")
    print("\nExemples (jusqu'à 10):")
    for d in decisions[:10]:
        print(f" - {d.src} -> {d.dest_rel if d.dest_rel else '-'} [{d.reason}] {d.human_size()}")

    # Tree preview
    print("\n=== APERÇU ARBORESCENCE CIBLE (simulée) ===")
    tree = build_tree(decisions)
    # Flatten paths
    all_paths = set()
    for parent, files in tree.items():
        cur = parent
        while True:
            all_paths.add(cur)
            if cur == Path('.'): break
            if not cur.parts:
                break
            cur = cur.parent
    # Sort paths depth-first
    for p in sorted(all_paths, key=lambda x: (len(x.parts), str(x))):
        indent = "  " * (len(p.parts)-1) if p.parts else ""
        print(f"{indent}{p}/")
        files = tree.get(p, [])
        for f in files[:5]:
            print(f"{indent}  {f.dest_rel.name} ({f.human_size()})")
        if len(files) > 5:
            print(f"{indent}  ... (+{len(files)-5} autres)")
    print()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str):
    ensure_dir(path.parent)
    path.write_text(content, encoding='utf-8')


def copy_files(repo_root: Path, decisions: List[FileDecision]):
    for d in decisions:
        if not d.will_copy or not d.dest_rel:
            continue
        dst = repo_root / d.dest_rel
        ensure_dir(dst.parent)
        shutil.copy2(d.src, dst)


def write_manifest(repo_root: Path, decisions: List[FileDecision]):
    lines = []
    lines.append(f"Généré: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Total ignorés: {sum(1 for d in decisions if not d.will_copy)}")
    lines.append("")
    for d in sorted((x for x in decisions if not x.will_copy), key=lambda x: x.src.as_posix()):
        lines.append(f"{d.src.as_posix()}\t{d.human_size()}\t{d.reason}")
    write_file(repo_root / SKIPPED_MANIFEST_REL, "\n".join(lines) + "\n")


def init_git(repo_root: Path):
    try:
        subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
        subprocess.run(["git", "commit", "-m", "chore: initial public drop (OpenHW 2025)"], cwd=repo_root, check=True)
    except FileNotFoundError:
        print("[WARN] git non trouvé dans le PATH – étape git ignorée.")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Échec git: {e}")


def attempt_github_push(repo_root: Path):
    gh_repo = os.environ.get("GITHUB_REPO")
    if not gh_repo:
        print("[INFO] Variable GITHUB_REPO non définie – push GitHub non tenté.")
        print("Astuce: export GITHUB_REPO=<User>/Tomo-AbandonedObject-FPGA")
        return
    # Vérifier gh
    try:
        r = subprocess.run(["gh", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("[WARN] CLI GitHub 'gh' introuvable – skip création remote.")
        return
    try:
        subprocess.run(["gh", "repo", "create", gh_repo, "--public", "--source=.", "--remote=origin", "--push"], cwd=repo_root, check=True)
        print(f"[OK] Dépôt GitHub créé et poussé: {gh_repo}")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Échec création repo GitHub ({gh_repo}): {e}")
        print("Commande manuelle suggérée après création remote:")
        print("  git remote add origin git@github.com:%s.git" % gh_repo)
        print("  git branch -M main")
        print("  git push -u origin main")


def generate_scaffold(repo_root: Path):
    # Répertoires clefs
    for d in [
        "report", "dataset/scripts", "training/runs", "training/notebooks",
        "gui", "fpga/vivado", "fpga/vitis", "fpga/hls", "docs", "tools"
    ]:
        ensure_dir(repo_root / d)
    # Fichiers principaux
    readme = repo_root / "README.md"
    if not readme.exists():
        write_file(readme, README_MD_TEMPLATE)
    write_file(repo_root / "LICENSE", LICENSE_TEXT)
    write_file(repo_root / ".gitignore", GITIGNORE_CONTENT)


def recap(decisions: List[FileDecision], repo_root: Path):
    copied = [d for d in decisions if d.will_copy]
    skipped = [d for d in decisions if not d.will_copy]
    total_size = sum(d.size for d in copied)
    print("\n=== RÉCAPITULATIF ===")
    print(f"Fichiers copiés : {len(copied)} | Taille totale : {human_readable_size(total_size)}")
    print(f"Fichiers ignorés: {len(skipped)} | Manifest: {repo_root / SKIPPED_MANIFEST_REL}")
    print("\nCommandes utiles:")
    print("python pack_tomo_repo.py --dry-run")
    print("python pack_tomo_repo.py --run")
    print("# Publication GitHub (exemple):")
    print("export GITHUB_REPO=<TonUser>/Tomo-AbandonedObject-FPGA")
    print(f"cd {REPO_DIR_NAME}")
    print("gh auth login  # si besoin")
    print("gh repo create \"$GITHUB_REPO\" --public --source=. --remote=origin --push")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prépare un dépôt propre Tomo-AbandonedObject-FPGA")
    p.add_argument("--run", action="store_true", help="Exécute réellement la copie (sinon dry-run)")
    p.add_argument("--dry-run", action="store_true", help="Force mode dry-run (par défaut si --run absent)")
    p.add_argument("--max-file-mb", type=int, default=DEFAULT_MAX_MB, help="Seuil taille en MB (def=95)")
    p.add_argument("--include-large", action="store_true", help="Tenter la copie même si > seuil")
    p.add_argument("--include-dataset", action="store_true", help="Inclure les dossiers train/ valid/ test/ (par défaut exclus)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.run:
        args.dry_run = True
    root = Path.cwd()
    print(f"[INFO] Racine workspace: {root}")
    decisions = scan_workspace(root, args)
    print_preview(decisions)

    repo_root = root / REPO_DIR_NAME
    if args.dry_run:
        print("[DRY-RUN] Aucune écriture effectuée. Utiliser --run pour exécuter.")
        # Simuler manifest path
        print(f"Manifest (sera généré): {repo_root / SKIPPED_MANIFEST_REL}")
        recap(decisions, repo_root)
        return 0

    # RUN mode
    print(f"[RUN] Création / mise à jour du dépôt: {repo_root}")
    generate_scaffold(repo_root)
    copy_files(repo_root, decisions)
    write_manifest(repo_root, decisions)
    init_git(repo_root)
    attempt_github_push(repo_root)
    recap(decisions, repo_root)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
