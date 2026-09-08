"""Gradio front-end for the cricket umpire signal classifier.

This file is deployed as-is to a Hugging Face Space. Everything it needs
(the exported model, the label helper, example images) lives alongside it
in this same folder, so the folder is self-contained.
"""

from pathlib import Path

import gradio as gr
from fastai.vision.all import load_learner

# `learn.export()` pickled a reference to `labeling.label_from_filename`
# (used to build the training DataLoaders). Unpickling the model needs that
# module importable, which is why labeling.py must sit next to this file
# even though we never call the function directly at inference time.
import labeling  # noqa: F401

MODEL_PATH = Path(__file__).parent / "umpire_signal_resnet34.pkl"
EXAMPLES_DIR = Path(__file__).parent / "examples"

learn = load_learner(MODEL_PATH)


def classify(img):
    if img is None:
        return None
    _, _, probs = learn.predict(img)
    return {label: float(prob) for label, prob in zip(learn.dls.vocab, probs)}


examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else None

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Umpire photo"),
    outputs=gr.Label(num_top_classes=5, label="Predicted signal"),
    title="Cricket Umpire Signal Classifier",
    description=(
        "Upload a photo of a cricket umpire and the model predicts which "
        "signal they're giving: Six, No ball, Out, Wide, or No action."
    ),
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()
