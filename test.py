# save as count_classes.py and run: python count_classes.py
import os
root = "./dataset"
if not os.path.isdir(root):
    print("Dataset folder not found:", root); raise SystemExit(1)
classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
print("Detected class folders:", classes)
for c in classes:
    files = [f for f in os.listdir(os.path.join(root, c)) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
    print(f"  {c}: {len(files)} files")
