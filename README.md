# Cricket Umpire Signal Classifier

Fine-tunes a pretrained CNN (fastai / ResNet) to classify which signal a
cricket umpire is giving from a single photo: **Six, No ball, Out, Wide, No
action**.

Dataset: [Cricket Umpires Action Classification](https://www.kaggle.com/datasets/warcoder/cricket-umpires-action-classification)
on Kaggle (a mirror of the SNOW dataset — 390 images, 78 per class).

## Project layout

```
data/umpire_poses/     the training images (class name is the filename prefix)
scripts/                all the Python code
models/                 trained .pkl models + confusion matrix / top-losses plots
.venv/                  project-local Python environment (created below)
```

## One-time setup

Requires Python 3.11+ already installed (`py --version` to check) and an
NVIDIA GPU if you want training to be fast.

1. **Create the virtual environment** — a private copy of Python + packages
   just for this project, so it can't clash with anything else on your
   machine:

   ```bash
   py -m venv .venv
   ```

2. **Install PyTorch first, from the CUDA 12.8 wheel index.** This step
   matters: a plain `pip install torch` can silently give you a build with no
   kernels for newer NVIDIA GPUs (Blackwell / RTX 50-series), which fails
   with `CUDA error: no kernel image is available for execution on the
   device`.

   ```bash
   .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install everything else:**

   ```bash
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

## Running the scripts

Every command below is run from the project root, calling the venv's own
`python.exe` directly by path. This works regardless of what `python` on your
system PATH points to (on Windows, the bare `python` command often resolves
to a non-functional Microsoft Store stub instead of a real install).

Alternatively, activate the venv once per terminal session and just use
`python`:

```bash
.venv\Scripts\Activate.ps1     # PowerShell
python scripts\check_gpu.py    # now `python` means the venv's python
```

### 1. Check the GPU is visible

Run this first, before anything else. Must print `CUDA available: True` and
your GPU's name.

```bash
.venv\Scripts\python.exe scripts\check_gpu.py
```

### 2. Get the data (only needed once)

Downloads the Kaggle dataset and flattens it into `data/umpire_poses/`.
Requires Kaggle API credentials (`kaggle.json` in `~/.kaggle/`, or
`KAGGLE_USERNAME`/`KAGGLE_KEY`/`KAGGLE_API_TOKEN` environment variables).

```bash
.venv\Scripts\python.exe scripts\download_kaggle.py
```

### 3. Verify the data

Prints per-class image counts and flags any class imbalance. Worth rerunning
any time you add or change images.

```bash
.venv\Scripts\python.exe scripts\verify_data.py
```

### 4. Train

Fine-tunes a pretrained CNN and writes the confusion matrix, a grid of the
most-wrong predictions, and the exported model into `models/`.

```bash
.venv\Scripts\python.exe scripts\train.py
```

Useful flags:

```bash
.venv\Scripts\python.exe scripts\train.py --arch resnet50 --epochs 10
```

| Flag | Default | Meaning |
|---|---|---|
| `--arch` | `resnet34` | `resnet34` or `resnet50` |
| `--epochs` | `6` | fine-tuning epochs after the initial frozen pass |
| `--valid-pct` | `0.2` | fraction of images held out for validation |
| `--bs` | `32` | batch size |
| `--img-size` | `224` | image resize before training |

Outputs land in `models/`:
- `umpire_signal_<arch>.pkl` — the exported model
- `confusion_matrix_<arch>.png` — which classes get mixed up with which
- `top_losses_<arch>.png` — the specific photos the model was most wrong about

## Notes

- The dataset is tiny (390 images across 5 classes), so accuracy varies
  noticeably between runs even with the same settings — judge the model by
  the confusion matrix, not a single accuracy number.
- Data loading uses `num_workers=0` because Windows can't share GPU memory
  across the worker processes PyTorch would otherwise spawn for loading —
  multi-worker loading crashes with `Cannot pickle CUDA storage`.
