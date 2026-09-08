"""Streamlit front-end for the cricket umpire signal classifier.

Deployed on Streamlit Community Cloud, which runs this file directly from
the GitHub repo. Everything it needs (the exported model, the label
helper, example images) lives alongside it in this same folder.
"""

import pathlib
import platform
from pathlib import Path

import pandas as pd
import streamlit as st
from fastai.vision.all import PILImage, load_learner

# `learn.export()` pickled a reference to `labeling.label_from_filename`
# (used to build the training DataLoaders). Unpickling the model needs that
# module importable, which is why labeling.py must sit next to this file
# even though we never call the function directly at inference time.
import labeling  # noqa: F401

# The model was exported on Windows, so fastai also pickled its training
# data path as a WindowsPath. Linux (Streamlit Cloud) refuses to instantiate
# WindowsPath, so unpickling fails with "cannot instantiate 'WindowsPath' on
# your system" unless we alias it to PosixPath first. Guarded so this is a
# no-op when testing locally on Windows.
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "umpire_signal_resnet34.pkl"
EXAMPLES_DIR = APP_DIR / "examples"

st.set_page_config(page_title="Cricket Umpire Signal Classifier", page_icon="🏏")


@st.cache_resource
def get_learner():
    return load_learner(MODEL_PATH)


learn = get_learner()

st.title("🏏 Cricket Umpire Signal Classifier")
st.write(
    "Upload a photo of a cricket umpire and the model predicts which signal "
    "they're giving: Six, No ball, Out, Wide, or No action."
)

if "chosen_image_bytes" not in st.session_state:
    st.session_state.chosen_image_bytes = None

uploaded = st.file_uploader("Umpire photo", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    st.session_state.chosen_image_bytes = uploaded.getvalue()

st.write("Or try an example:")
example_paths = sorted(EXAMPLES_DIR.glob("*.jpg"))
cols = st.columns(len(example_paths))
for col, path in zip(cols, example_paths):
    with col:
        st.image(str(path), width="stretch")
        if st.button("Use this", key=f"example-{path.name}"):
            st.session_state.chosen_image_bytes = path.read_bytes()

if st.session_state.chosen_image_bytes:
    img = PILImage.create(st.session_state.chosen_image_bytes)
    st.image(img, caption="Selected image", width=300)

    with st.spinner("Classifying..."):
        _, pred_idx, probs = learn.predict(img)

    pred_label = learn.dls.vocab[int(pred_idx)]
    st.subheader(f"Prediction: {pred_label}")

    probs_series = pd.Series(
        {label: float(prob) for label, prob in zip(learn.dls.vocab, probs)},
        name="probability",
    ).sort_values(ascending=False)
    st.bar_chart(probs_series)
