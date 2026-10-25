# Skrip pelatihan ConvNeXt-Tiny (timm, pretrained IN-22k -> IN-1k) untuk klasifikasi sampah 5 kelas

import argparse
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.utils import ModelEmaV3
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

# Urutan kelas sama dengan ImageFolder (alfabetis)
CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
NUM_CLASSES = 5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


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


def build_model(model_name, drop_path):
    # ConvNeXt-Tiny timm dengan bobot pretrained, head diganti 5 kelas.
    # Key state_dict sama dengan ConvNeXtTTA di submission/model.py
    return timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES,
                             drop_path_rate=drop_path)


def get_train_transform(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(0.2),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])


def get_val_transform(img_size):
    # Harus sama dengan get_transform() di submission/model.py
    return transforms.Compose([
        transforms.Resize(img_size + 16),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
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
def predict(model, loader, device):
    # Prediksi dengan TTA flip horizontal, sama seperti ConvNeXtTTA saat eval
    model.eval()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = (model(images) + model(torch.flip(images, dims=[3]))) / 2
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())

    return y_true, y_pred


def evaluate(model, loader, device):
    return compute_macro_f1(*predict(model, loader, device))


def train_one_epoch(model, ema, loader, criterion, optimizer, scheduler, scaler, mixer, mix_prob, device):
    model.train()
    total_loss = 0
    n = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # MixUp / CutMix pada sebagian batch, label menjadi soft target
        if mixer is not None and random.random() < mix_prob:
            images, labels = mixer(images, labels)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    return total_loss / n


def build_optimizer(model, lr, head_lr, weight_decay):
    # Head baru pakai LR lebih besar; bias, norm, dan layer scale tanpa weight decay
    groups = {}
    for name, p in model.named_parameters():
        is_head = name.startswith("head.fc")
        no_wd = p.ndim <= 1
        key = (is_head, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": head_lr if is_head else lr,
                           "weight_decay": 0.0 if no_wd else weight_decay}
        groups[key]["params"].append(p)
    return AdamW(list(groups.values()))


def save_fp16(model, path):
    # Simpan dalam fp16 agar < 100 MB (batas GitHub)
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./train", help="folder dataset train")
    parser.add_argument("--model-name", default="convnext_tiny.fb_in22k_ft_in1k", help="nama model timm")
    parser.add_argument("--img-size", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=1, help="epoch warmup linear")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="LR backbone")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="LR head")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--mix-prob", type=float, default=0.5, help="peluang MixUp/CutMix per batch, 0 = mati")
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--full", action="store_true", help="latih di my_train + my_val, simpan EMA epoch terakhir")
    parser.add_argument("--dump-errors", action="store_true", help="cetak gambar my_val yang salah prediksi")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Buat split lokal
    split_dataset(args.data, "./my_train", "./my_val")

    # Dataset
    train_tf = get_train_transform(args.img_size)
    train_ds = datasets.ImageFolder("./my_train", transform=train_tf)
    val_ds = datasets.ImageFolder("./my_val", transform=get_val_transform(args.img_size))
    if args.full:
        train_ds = ConcatDataset([train_ds, datasets.ImageFolder("./my_val", transform=train_tf)])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}  (full={args.full})")
    print(f"Kelas: {val_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Bangun model dengan bobot pretrained
    model = build_model(args.model_name, args.drop_path).to(device)
    ema = ModelEmaV3(model, decay=args.ema_decay, use_warmup=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(model, args.lr, args.head_lr, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # Warmup linear lalu cosine, diupdate per step
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    mixer = None
    if args.mix_prob > 0:
        mixer = v2.RandomChoice([v2.MixUp(alpha=0.2, num_classes=NUM_CLASSES),
                                 v2.CutMix(alpha=1.0, num_classes=NUM_CLASSES)])

    pth = Path("submission/model.pth")
    best_f1, best_epoch = 0.0, 0
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, ema, train_loader, criterion, optimizer, scheduler,
                               scaler, mixer, args.mix_prob, device)
        f1, acc, per_cls = evaluate(ema.module, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{args.epochs}  loss={loss:.4f}  EMA F1={f1:.4f}  acc={acc:.4f}  "
              f"kertas={per_cls['kertas_kardus']:.4f}  lr={lr_now:.6f}")

        # Mode validasi: simpan EMA terbaik. Mode full: my_val ikut dilatih, jadi simpan epoch terakhir
        if not args.full and f1 > best_f1:
            best_f1, best_epoch = f1, epoch + 1
            save_fp16(ema.module, pth)
            print(f"    -> Simpan checkpoint (F1={f1:.4f})")

    if args.full:
        save_fp16(ema.module, pth)
        print(f"\nSelesai (full). EMA epoch terakhir disimpan, F1 my_val (bukan held-out) = {f1:.4f}")
    else:
        print(f"\nSelesai. Best Macro-F1 = {best_f1:.4f} di epoch {best_epoch}")

    if args.dump_errors and not args.full:
        # Muat ulang checkpoint terbaik lalu cetak gambar yang salah prediksi
        ema.module.load_state_dict(torch.load(pth, map_location=device, weights_only=True))
        y_true, y_pred = predict(ema.module, val_loader, device)
        print("\nGambar my_val yang salah prediksi (cek kemungkinan label noise):")
        for (path, _), t, p in zip(val_ds.samples, y_true, y_pred):
            if t != p:
                print(f"  {path}  label={CLASSES[t]}  pred={CLASSES[p]}")

    print("Jalankan evaluasi: python evaluate.py --submission ./submission --data ./my_val")
    size_mb = pth.stat().st_size / (1024 * 1024)
    print(f"Ukuran model.pth (fp16): {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
