import os, shutil

base = r"C:\Dsproject\dataset"
for cls in os.listdir(base):
    outer = os.path.join(base, cls)
    if not os.path.isdir(outer): continue
    # find nested folder with same name
    inner = os.path.join(outer, cls)
    if os.path.isdir(inner):
        for f in os.listdir(inner):
            src = os.path.join(inner, f)
            dst = os.path.join(outer, f)
            if os.path.isfile(src):
                shutil.move(src, dst)
        shutil.rmtree(inner)
        print(f"Flattened nested folder for class: {cls}")

print("✅ Done. Your dataset folder structure is now clean.")
