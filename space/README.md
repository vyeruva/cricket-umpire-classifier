---
title: Cricket Umpire Signal Classifier
emoji: 🏏
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# Cricket Umpire Signal Classifier

Upload a photo of a cricket umpire and this model predicts which signal
they're giving: **Six, No ball, Out, Wide, No action**.

Fine-tuned resnet34, trained with fastai on the
[Cricket Umpires Action Classification](https://www.kaggle.com/datasets/warcoder/cricket-umpires-action-classification)
dataset (390 images, 78 per class).
