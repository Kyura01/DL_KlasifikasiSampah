# Submission Studio 2: ConvNeXt-Tiny (timm), head 5 kelas, TTA flip horizontal.
#
# get_model()     -> arsitektur saja (tanpa bobot), evaluate.py yang memuat bobot
# get_transform() -> transform validasi (sama dengan get_val_transform di train.py)
#
# Urutan kelas (ImageFolder, alfabetis):
#   0=botol_plastik, 1=gelas_plastik, 2=kertas_kardus, 3=saset_kemasan, 4=styrofoam

import torch
from timm.models.convnext import ConvNeXt
from torchvision import transforms

NUM_CLASSES = 5
IMG_SIZE = 288
RESIZE = IMG_SIZE + 16
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ConvNeXtTTA(ConvNeXt):
    # ConvNeXt-Tiny biasa; saat eval logits dirata-rata dengan versi flip horizontal.
    # Key state_dict identik dengan timm convnext_tiny, jadi strict=True tetap lolos.
    def forward(self, x):
        logits = super().forward(x)
        if not self.training:
            logits = (logits + super().forward(torch.flip(x, dims=[3]))) / 2
        return logits


def get_model():
    return ConvNeXtTTA(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), num_classes=NUM_CLASSES)


def get_transform():
    return transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
