# Skrip pelatihan ConvNeXt-Tiny untuk klasifikasi sampah 5 kelas

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights

# Urutan kelas sama dengan ImageFolder (alfabetis)
CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
NUM_CLASSES = 5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(src, dst_train, dst_val, val_ratio=0.15, seed=42):
    # Buat split stratified dari src ke dst_train dan dst_val
    if Path(dst_train).exists() and Path(dst_val).exists():
        print("Split sudah ada, skip pembuatan split.")
        return

    print(f"Membuat split dari {src} ...")
    rng = random.Random(seed)

    for cls in sorted(os.listdir(src)):
        cls_dir = Path(src) / cls
        if not cls_dir.is_dir():
            continue

        files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])
        rng.shuffle(files)

        n_val = max(1, int(len(files) * val_ratio))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for f in train_files:
            dst = Path(dst_train) / cls
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f.name)

        for f in val_files:
            dst = Path(dst_val) / cls
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f.name)

        print(f"  {cls}: {len(train_files)} train, {len(val_files)} val")

    print("Split selesai.\n")


def build_model(pretrained=True):
    # Bangun ConvNeXt-Tiny, opsional dengan bobot pretrained
    if pretrained:
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
    else:
        model = models.convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def compute_macro_f1(y_true, y_pred, k=NUM_CLASSES):
    # Hitung macro F1, sama seperti di evaluate.py
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    f1s = []
    for c in range(k):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(k)) - tp
        fn = sum(cm[c]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)

    macro_f1 = sum(f1s) / k
    acc = sum(cm[c][c] for c in range(k)) / max(1, len(y_true))
    return macro_f1, acc, dict(zip(CLASSES, f1s))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

    return compute_macro_f1(y_true, y_pred)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./train", help="folder dataset train")
    parser.add_argument("--epochs1", type=int, default=3, help="epoch tahap 1 (head only)")
    parser.add_argument("--epochs2", type=int, default=10, help="epoch tahap 2 (full)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Buat split lokal
    split_dataset(args.data, "./my_train", "./my_val")

    # Dataset
    train_ds = datasets.ImageFolder("./my_train", transform=get_train_transform())
    val_ds = datasets.ImageFolder("./my_val", transform=get_val_transform())
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Kelas: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Bangun model dengan bobot pretrained
    model = build_model(pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # === Tahap 1: hanya head ===
    print("\n--- Tahap 1: latih head saja ---")

    # Bekukan backbone
    for param in model.features.parameters():
        param.requires_grad = False

    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer1 = AdamW(head_params, lr=1e-3, weight_decay=0.05)

    best_f1 = 0.0
    for epoch in range(args.epochs1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer1, device)
        f1, acc, per_cls = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{args.epochs1}  loss={loss:.4f}  F1={f1:.4f}  acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "submission/model.pth")
            print(f"    -> Simpan checkpoint (F1={f1:.4f})")

    # === Tahap 2: semua layer ===
    print("\n--- Tahap 2: fine-tune semua layer ---")

    # Buka semua parameter
    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer2, T_max=args.epochs2)

    for epoch in range(args.epochs2):
        loss = train_one_epoch(model, train_loader, criterion, optimizer2, device)
        scheduler.step()
        f1, acc, per_cls = evaluate(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{args.epochs2}  loss={loss:.4f}  F1={f1:.4f}  acc={acc:.4f}  lr={lr_now:.6f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "submission/model.pth")
            print(f"    -> Simpan checkpoint (F1={f1:.4f})")

    print(f"\nSelesai. Best Macro-F1 = {best_f1:.4f}")
    print("Jalankan evaluasi: python evaluate.py --submission ./submission --data ./my_val")

    # Cek ukuran model.pth
    pth = Path("submission/model.pth")
    size_mb = pth.stat().st_size / (1024 * 1024)
    print(f"Ukuran model.pth: {size_mb:.1f} MB")

    if size_mb > 95:
        print("Model > 95 MB, simpan ulang dalam fp16 ...")
        state = torch.load(pth, map_location="cpu", weights_only=True)
        state_fp16 = {k: v.half() for k, v in state.items()}
        torch.save(state_fp16, pth)
        size_mb = pth.stat().st_size / (1024 * 1024)
        print(f"Ukuran model.pth (fp16): {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
