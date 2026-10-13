# Submission Studio 2: ConvNeXt-Tiny, head 5 kelas.
#
# get_model()     -> arsitektur saja (weights=None), evaluate.py yang memuat bobot
# get_transform() -> transform validasi (sama dengan train.py)
#
# Urutan kelas (ImageFolder, alfabetis):
#   0=botol_plastik, 1=gelas_plastik, 2=kertas_kardus, 3=saset_kemasan, 4=styrofoam

import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
IMG_SIZE = 224
RESIZE = 236
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_model():
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
