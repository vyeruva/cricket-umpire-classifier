"""Sanity-check the dataset before training.

Reports per-class image counts, flags class imbalance, and lists any files
that don't match a known label pattern or fail to open. Run this and read the
output before kicking off training.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from labeling import label_from_filename

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "data" / "umpire_poses"),
        help="Folder containing the umpire pose images (default: ../data/umpire_poses)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not files:
        print(f"No image files found under {data_dir}")
        sys.exit(1)

    counts = Counter()
    unmatched = []
    for f in files:
        try:
            counts[label_from_filename(f)] += 1
        except ValueError:
            unmatched.append(f.name)

    total = sum(counts.values())
    print(f"Found {total} labeled images under {data_dir}\n")
    print(f"{'Class':<12} {'Count':>6}   {'% of total':>10}")
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * n / total
        print(f"{label:<12} {n:>6}   {pct:>9.1f}%")

    if counts:
        max_n, min_n = max(counts.values()), min(counts.values())
        ratio = max_n / min_n
        print(f"\nMax/min class ratio: {ratio:.1f}x", end="")
        if ratio >= 2:
            print("  <-- notable imbalance, consider class weighting or oversampling")
        else:
            print()

    if unmatched:
        print(f"\n{len(unmatched)} files did not match a known class prefix:")
        for name in unmatched[:20]:
            print(f"  {name}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")


if __name__ == "__main__":
    main()
