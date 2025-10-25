# model_timm.py
import os
from typing import List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageFolderAlb(Dataset):
    """
    Simple ImageFolder-like dataset that uses Albumentations transforms.
    Expects directory structure: root/<class_name>/*.jpg
    """
    def __init__(self, samples: List[Tuple[str,int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB")
        img = np.array(img)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        return img, label

def make_samples_from_folder(root: str):
    root = Path(root)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    samples = []
    for cls_idx, cls_name in enumerate(classes):
        for f in (root/cls_name).iterdir():
            if f.suffix.lower() in ('.jpg','.jpeg','.png',''):
                samples.append((str(f), cls_idx))
    return samples, classes

def get_transforms(input_size=300, mode='train'):
    """
    Robust Albumentations transforms that try multiple constructor signatures
    for RandomResizedCrop / Resize to support many albumentations versions.
    """
    from albumentations.pytorch import ToTensorV2

    h = input_size
    w = input_size

    # helper to build RandomResizedCrop with multiple possible signatures
    def build_random_resized_crop(h, w, scale=(0.7, 1.0), ratio=(0.75, 1.333), p=1.0):
        # try tuple 'size' keyword
        try:
            return A.RandomResizedCrop(size=(h, w), scale=scale, ratio=ratio, p=p)
        except Exception:
            pass
        # try positional (h, w)
        try:
            return A.RandomResizedCrop(h, w, scale=scale, ratio=ratio, p=p)
        except Exception:
            pass
        # try single int positional (some versions expect a single int -> tuple)
        try:
            return A.RandomResizedCrop(int(h), scale=scale, ratio=ratio, p=p)
        except Exception:
            pass
        # fallback: RandomCrop + Resize (compose to mimic random resized crop)
        try:
            return A.Compose([
                A.RandomCrop(height=h, width=w, p=1.0),
                A.Resize(h, w)
            ])
        except Exception:
            # last-ditch simple transform (should rarely reach here)
            return A.RandomCrop(height=h, width=w, p=1.0)

    # helper to build Resize with multiple signatures
    def build_resize(h, w):
        try:
            return A.Resize(height=h, width=w)
        except Exception:
            pass
        try:
            return A.Resize(h, w)
        except Exception:
            pass
        try:
            return A.Resize(size=(h, w))
        except Exception:
            pass
        # fallback - CenterCrop + Resize where available
        try:
            return A.Compose([A.CenterCrop(h, w), A.Resize(h, w)])
        except Exception:
            # final fallback: No-op (may be handled later)
            return A.NoOp()

    if mode == 'train':
        rand_crop = build_random_resized_crop(h, w, scale=(0.7, 1.0), ratio=(0.75, 1.333), p=1.0)
        return A.Compose([
            rand_crop,
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            A.OneOf([A.Blur(blur_limit=3), A.GaussNoise(var_limit=(5.0, 20.0))], p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        resize = build_resize(h, w)
        return A.Compose([
            resize,
            A.Normalize(),
            ToTensorV2(),
        ])


def build_model_timm(model_name='tf_efficientnet_b3', num_classes=2, pretrained=True, drop_rate=0.2):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    return model

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location=None):
    if map_location is None:
        map_location = DEVICE
    return torch.load(path, map_location=map_location)

def tta_predict_batch(models: List[nn.Module], pil_img: Image.Image, input_size:int=300, tta_transforms: List = None):
    """
    Simple TTA: apply flips and center-crop/rescale. `models` are already on DEVICE and eval()
    Returns averaged softmax probabilities.
    """
    if tta_transforms is None:
        # default: center + horizontal flip
        tta_transforms = [
            A.Compose([A.Resize(input_size, input_size), A.Normalize(), ToTensorV2()]),
            A.Compose([A.HorizontalFlip(p=1.0), A.Resize(input_size, input_size), A.Normalize(), ToTensorV2()]),
        ]
    arr = np.array(pil_img.convert("RGB"))
    probs_acc = None
    with torch.no_grad():
        for tf in tta_transforms:
            x = tf(image=arr)['image'].unsqueeze(0).to(DEVICE)
            # average across models
            model_probs = None
            for m in models:
                out = m(x)
                p = torch.softmax(out, dim=1).cpu().numpy()
                if model_probs is None:
                    model_probs = p
                else:
                    model_probs += p
            model_probs /= len(models)
            if probs_acc is None:
                probs_acc = model_probs
            else:
                probs_acc += model_probs
    probs_acc /= len(tta_transforms)
    return probs_acc[0].tolist()
