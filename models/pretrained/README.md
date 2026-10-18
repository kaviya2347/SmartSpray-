# Pretrained backbone weights

Put the official Keras ImageNet weights for EfficientNetB0 (no top) here:

    models/pretrained/efficientnetb0_notop.h5

Download link (≈16 MB, official Keras/Google bucket):
https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5

`scripts/m3_train_classifier.py` loads this file when present, so training
works offline. If the file is missing, Keras downloads it automatically
(requires internet access).
