# train.py
"""
CLI training script.

Examples:
  # Download dataset (optional) then train:
  python train.py --data ./dataset --epochs 8 --batch 32 --lr 1e-4 --model_out best_model.pth --pretrained

If you want automatic KaggleHub download:
  python train.py --download_kaggle --kaggle_name cashbowman/ai-generated-images-vs-real-images
"""
import argparse
import os
from model import create_dataloaders, build_model, train
import sys

def try_download_kagglehub(dataset_name: str, out_dir: str = "dataset"):
    try:
        import kagglehub
    except Exception as e:
        print("kagglehub not installed or failed to import:", e)
        return False
    try:
        print("Downloading dataset via kagglehub:", dataset_name)
        path = kagglehub.dataset_download(dataset_name)
        # kagglehub may return a folder path; if not, user should move/extract to out_dir
        if path and os.path.exists(path):
            # If path is not './dataset', attempt to copy or symlink
            if os.path.abspath(path) != os.path.abspath(out_dir):
                print("Copying downloaded dataset to", out_dir)
                try:
                    import shutil
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir)
                    shutil.copytree(path, out_dir)
                except Exception as e:
                    print("Copy failed (you may need to move files manually):", e)
            return True
        return False
    except Exception as e:
        print("kagglehub download failed:", e)
        return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root (ImageFolder).')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--model_out', type=str, default='best_model.pth')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--download_kaggle', action='store_true', help='Try to download dataset using kagglehub before training.')
    parser.add_argument('--kaggle_name', type=str, default='cashbowman/ai-generated-images-vs-real-images', help='Kaggle dataset slug for download (kagglehub).')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.download_kaggle:
        ok = try_download_kagglehub(args.kaggle_name, out_dir=args.data)
        if not ok:
            print("Kaggle download failed. Make sure kagglehub is installed and you have permissions. You can download manually and place files into", args.data)
    if not os.path.exists(args.data):
        print("Dataset path not found:", args.data)
        print("Please download dataset and ensure structure: dataset/<CLASS_NAME>/*.jpg (e.g., dataset/AI, dataset/REAL).")
        sys.exit(1)

    print("Creating dataloaders...")
    train_loader, val_loader, class_names = create_dataloaders(args.data, input_size=224, batch_size=args.batch, val_split=args.val_split)
    print("Classes:", class_names)
    model = build_model(num_classes=len(class_names), pretrained=args.pretrained)
    print("Starting training...")
    best_acc = train(model, train_loader, val_loader, class_names, epochs=args.epochs, lr=args.lr, model_path=args.model_out)
    print("Training finished. Best val acc:", best_acc)

if __name__ == '__main__':
    main()
