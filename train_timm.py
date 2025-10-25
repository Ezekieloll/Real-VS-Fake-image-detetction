# train_timm.py
"""
Train script with timm backbone, Albumentations, MixUp, OneCycleLR, StratifiedKFold ensemble.

Usage examples:
  # single-run (no kfold)
  python train_timm.py --data ./dataset --model_out best.pth --model_name tf_efficientnet_b3 --epochs 12 --batch 16 --input_size 300 --pretrained

  # 4-fold ensemble
  python train_timm.py --data ./dataset --kfold 4 --model_name tf_efficientnet_b3 --epochs 12 --batch 12 --input_size 300 --pretrained
"""
import argparse
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import random
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model_timm import make_samples_from_folder, ImageFolderAlb, get_transforms, build_model_timm, save_checkpoint, DEVICE

# MixUp helper
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, use_mixup=False, mixup_alpha=0.4, scheduler=None):
    model.train()
    running_loss = 0.0
    preds, targets = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if use_mixup:
            imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
            outputs = model(imgs)
            if use_mixup:
                loss = lam * loss_fn(outputs, targets_a) + (1-lam) * loss_fn(outputs, targets_b)
            else:
                loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass
        running_loss += loss.item() * imgs.size(0)
        preds.extend(torch.argmax(outputs.detach().cpu(), dim=1).numpy())
        targets.extend(labels.detach().cpu().numpy())
    epoch_loss = running_loss / max(1, len(loader.sampler))
    acc = (np.array(preds) == np.array(targets)).mean() if len(targets)>0 else 0.0
    return epoch_loss, acc

def eval_model(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds.extend(torch.argmax(outputs.detach().cpu(), dim=1).numpy())
            targets.extend(labels.detach().cpu().numpy())
    epoch_loss = running_loss / max(1, len(loader.sampler))
    acc = (np.array(preds) == np.array(targets)).mean() if len(targets)>0 else 0.0
    return epoch_loss, acc

def create_loader_from_indices(samples, indices, transform, batch_size, shuffle=False, num_workers=4):
    sub = [samples[i] for i in indices]
    ds = ImageFolderAlb(sub, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--model_name', type=str, default='tf_efficientnet_b3')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--input_size', type=int, default=300)
    p.add_argument('--model_out', type=str, default='best_model.pth')
    p.add_argument('--kfold', type=int, default=0, help='If >0, runs k-fold training and saves fold_i.pth files')
    p.add_argument('--mixup', action='store_true', help='Use MixUp')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_fold(samples, classes, args, fold_idx=None, train_idx=None, val_idx=None):
    device = DEVICE
    model = build_model_timm(model_name=args.model_name, num_classes=len(classes), pretrained=args.pretrained)
    model = model.to(device)
    # transforms
    train_tf = get_transforms(args.input_size, mode='train')
    val_tf = get_transforms(args.input_size, mode='val')
    train_loader = create_loader_from_indices(samples, train_idx, train_tf, batch_size=args.batch, shuffle=True)
    val_loader = create_loader_from_indices(samples, val_idx, val_tf, batch_size=args.batch, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # OneCycleLR
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_val_acc = 0.0
    best_path = None
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, use_mixup=args.mixup, mixup_alpha=0.4, scheduler=scheduler)
        val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
        print(f"Fold {fold_idx} Epoch {epoch}/{args.epochs} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.kfold > 0 and fold_idx is not None:
                out = f"model_fold{fold_idx}.pth"
            else:
                out = args.model_out
            save_checkpoint({
                "model_state_dict": model.state_dict(),
                "class_names": classes,
                "input_size": args.input_size,
                "model_name": args.model_name,
            }, out)
            best_path = out
            print(f"Saved best model to {out} (acc={best_val_acc:.4f})")
    return best_val_acc, best_path

def main():
    args = parse_args()
    set_seed(args.seed)
    samples, classes = make_samples_from_folder(args.data)
    if len(samples) == 0:
        print("No samples found under", args.data)
        return
    labels = [s[1] for s in samples]
    # K-Fold branch: requires each class to have at least `kfold` samples
    if args.kfold and args.kfold > 1:
        # quick check: ensure every class has >= kfold samples
        counts = Counter(labels)
        insufficient = [c for c, cnt in counts.items() if cnt < args.kfold]
        if insufficient:
            print("ERROR: Cannot run StratifiedKFold with kfold=%d because some classes have fewer than %d samples." % (args.kfold, args.kfold))
            print("Class counts:", dict(counts))
            print("Classes with insufficient samples:", insufficient)
            print("Either reduce --kfold or add more samples to those classes.")
            return

        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        fold = 0
        paths = []
        for train_idx, val_idx in skf.split(np.arange(len(samples)), labels):
            print(f"=== Fold {fold} ===")
            best_acc, best_path = train_fold(samples, classes, args, fold_idx=fold, train_idx=train_idx, val_idx=val_idx)
            paths.append(best_path)
            fold += 1
        print("K-Fold finished. Saved models:", paths)
    else:
        # single run: attempt a stratified split, but fall back to random split if impossible
        from sklearn.model_selection import train_test_split

        # compute counts per class (diagnostic)
        label_counts = Counter(labels)
        print("Label counts:", dict(label_counts))

        min_count = min(label_counts.values())
        if min_count < 2:
            print("WARNING: Some classes have fewer than 2 samples; stratified split is not possible.")
            print("Performing a random (non-stratified) split. Recommended: fix dataset so each class has >=2 samples.")
            train_idx, val_idx = train_test_split(np.arange(len(samples)), test_size=0.15, random_state=args.seed, shuffle=True)
        else:
            # safe to stratify
            train_idx, val_idx = train_test_split(np.arange(len(samples)), stratify=labels, test_size=0.15, random_state=args.seed)
        print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
        best_acc, best_path = train_fold(samples, classes, args, fold_idx=None, train_idx=train_idx, val_idx=val_idx)
        print("Training finished. Best model:", best_path, "val_acc:", best_acc)


if __name__ == '__main__':
    main()
