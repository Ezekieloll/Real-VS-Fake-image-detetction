# model.py
"""
Model utilities for Fake Image Detector:
- create_dataloaders
- build_model
- train (training loop saves best checkpoint)
- save_checkpoint / load_checkpoint
- predict_image (single PIL image)
"""
import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataloaders(data_dir: str, input_size: int = 224, batch_size: int = 32,
                       val_split: float = 0.15, seed: int = 42) -> Tuple[DataLoader, DataLoader, List[str]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(input_size*1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_tf)
    num = len(full_dataset)
    indices = list(range(num))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = int(np.floor(val_split * num))
    if split == 0:
        split = max(1, int(0.01 * num))
    train_idx, val_idx = indices[split:], indices[:split]

    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_dataset = datasets.ImageFolder(root=str(data_dir), transform=val_tf)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    class_names = full_dataset.classes
    return train_loader, val_loader, class_names

def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location=None) -> dict:
    if map_location is None:
        map_location = DEVICE
    ck = torch.load(path, map_location=map_location)
    return ck

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          class_names: List[str],
          epochs: int = 6,
          lr: float = 1e-4,
          weight_decay: float = 1e-4,
          model_path: str = "best_model.pth",
          device: torch.device = DEVICE):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_val_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        preds_train, targets_train = [], []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            preds_train.extend(torch.argmax(outputs.detach().cpu(), dim=1).numpy())
            targets_train.extend(labels.detach().cpu().numpy())
        train_loss = running_loss / max(1, len(train_loader.sampler))
        train_acc = accuracy_score(targets_train, preds_train) if len(targets_train) else 0.0

        model.eval()
        running_loss = 0.0
        preds_val, targets_val = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                preds_val.extend(torch.argmax(outputs.detach().cpu(), dim=1).numpy())
                targets_val.extend(labels.detach().cpu().numpy())
        val_loss = running_loss / max(1, len(val_loader.sampler))
        val_acc = accuracy_score(targets_val, preds_val) if len(targets_val) else 0.0

        scheduler.step(val_acc)
        print(f"Epoch {epoch}/{epochs} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": getattr(train_loader.dataset.transform, 'size', 224) if hasattr(train_loader.dataset.transform, 'size') else 224,
            }, model_path)
            print(f"Saved best model (acc={best_val_acc:.4f}) to {model_path}")

    return best_val_acc

def predict_image(model: nn.Module, pil_img: Image.Image, input_size: int = 224):
    model = model.to(DEVICE)
    model.eval()
    tf = transforms.Compose([
        transforms.Resize(int(input_size*1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img_t = tf(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_t)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0].tolist()
    return probs
