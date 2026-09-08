"""Fine-tune a pretrained CNN to classify cricket umpire signals.

Usage:
    python train.py                         # resnet34, defaults below
    python train.py --arch resnet50 --epochs 8

Run scripts/check_gpu.py and scripts/verify_data.py first.
"""

import argparse
from pathlib import Path

import matplotlib

# Headless box, no working Tk/display — write plots straight to file instead
# of trying to pop up an interactive window.
matplotlib.use("Agg")

from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    aug_transforms,
    vision_learner,
    error_rate,
    accuracy,
    ClassificationInterpretation,
    resnet34,
    resnet50,
)

from labeling import label_from_filename

ARCHES = {"resnet34": resnet34, "resnet50": resnet50}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "umpire_poses"),
    )
    parser.add_argument("--arch", choices=list(ARCHES), default="resnet34")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--valid-pct", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "models"),
        help="Directory to write the exported .pkl and diagnostic plots",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dls = ImageDataLoaders.from_name_func(
        data_dir,
        list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.png")),
        label_func=label_from_filename,
        valid_pct=args.valid_pct,
        seed=args.seed,
        bs=args.bs,
        # Windows can't fork worker processes with CUDA memory already
        # mapped in the parent, so multi-worker loading crashes with
        # "Cannot pickle CUDA storage". Single-process loading is plenty
        # fast for a dataset this small anyway.
        num_workers=0,
        item_tfms=Resize(args.img_size),
        # Small dataset -> lean hard on augmentation.
        batch_tfms=aug_transforms(
            mult=1.0,
            do_flip=True,
            flip_vert=False,
            max_rotate=15.0,
            max_zoom=1.15,
            max_lighting=0.3,
            max_warp=0.1,
        ),
    )

    print("Classes:", dls.vocab)
    print("Train/valid sizes:", len(dls.train_ds), len(dls.valid_ds))

    learn = vision_learner(dls, ARCHES[args.arch], metrics=[accuracy, error_rate])
    learn.fine_tune(args.epochs)

    interp = ClassificationInterpretation.from_learner(learn)

    cm_path = out_dir / f"confusion_matrix_{args.arch}.png"
    interp.plot_confusion_matrix(figsize=(7, 7), dpi=100)
    import matplotlib.pyplot as plt

    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    print("\nMost confused (actual, predicted, count):")
    for actual, predicted, n in interp.most_confused(min_val=1):
        print(f"  {actual:<12} -> {predicted:<12} : {n}")

    top_losses_path = out_dir / f"top_losses_{args.arch}.png"
    interp.plot_top_losses(9, nrows=3)
    plt.savefig(top_losses_path, bbox_inches="tight")
    plt.close()
    print(f"Saved top losses to {top_losses_path}")

    model_path = out_dir / f"umpire_signal_{args.arch}.pkl"
    learn.export(model_path)
    print(f"Exported model to {model_path}")


if __name__ == "__main__":
    main()
