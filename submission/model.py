"""Submission Studio 2: ConvNeXt-Tiny dengan head 5 kelas.

Kontrak (lihat evaluate.py):
  get_model()     hanya MEMBANGUN arsitektur. Tidak mengunduh bobot pretrained
                  dan tidak memuat model.pth, karena evaluate.py yang memuatnya
                  dengan load_state_dict(strict=True).
  get_transform() transform validasi, harus SAMA dengan yang dipakai di train.py.

Output model: logits [batch, 5] dengan urutan kelas alfabetis
(sama dengan torchvision.datasets.ImageFolder):
  0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
IMG_SIZE = 224
RESIZE = 236  # sama dengan preset ConvNeXt_Tiny_Weights.IMAGENET1K_V1
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_model():
    model = models.convnext_tiny(weights=None)
    # classifier = Sequential(LayerNorm2d, Flatten, Linear(768, 1000))
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
