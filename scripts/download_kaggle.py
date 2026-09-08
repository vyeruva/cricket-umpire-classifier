"""Download the SNOW dataset mirror from Kaggle and flatten it into
data/umpire_poses/ so filenames carry the class prefix (matches labeling.py).

Requires Kaggle auth via env vars (KAGGLE_USERNAME/KAGGLE_KEY) or a
kaggle.json in ~/.kaggle/, or a modern single-token KAGGLE_API_TOKEN, per
whatever the installed `kaggle` package version expects.
"""

import shutil
import zipfile
from pathlib import Path

DATASET = "warcoder/cricket-umpires-action-classification"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "_kaggle_raw"
DEST_DIR = PROJECT_ROOT / "data" / "umpire_poses"


def main():
    import kaggle

    kaggle.api.authenticate()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET} ...")
    kaggle.api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=False)

    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No zip found in {RAW_DIR} after download")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(RAW_DIR)
    print(f"Extracted to {RAW_DIR}")

    n = 0
    for img in RAW_DIR.rglob("*"):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dest = DEST_DIR / img.name
            if dest.exists():
                dest = DEST_DIR / f"{img.stem}_{n}{img.suffix}"
            shutil.copy2(img, dest)
            n += 1
    print(f"Copied {n} images into {DEST_DIR}")


if __name__ == "__main__":
    main()
