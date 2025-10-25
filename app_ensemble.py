# app_ensemble.py
import streamlit as st
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import glob
import traceback

import torch
import timm
import torchvision.models as tvmodels

from model_timm import build_model_timm, load_checkpoint, tta_predict_batch, DEVICE

st.set_page_config(page_title='Fake Image Detector - Ensemble Inference', layout='centered')
st.title('🖼️ Fake Image Detector — Ensemble Inference')

MODEL_PATTERN = st.text_input("Model path or wildcard (e.g., best_model.pth OR model_fold*.pth)", value="model_fold*.pth")
input_size = st.number_input("Input size (should match training)", value=300, step=32)

# ---------------- Robust loader helpers ----------------
def strip_module_prefix(state):
    """Strip 'module.' prefix from keys coming from DataParallel."""
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state

def try_timm_load(path, state, class_names, model_name):
    """Try to create a timm model and load state_dict (strict then non-strict)."""
    try:
        st = st = state
        model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names) if class_names else None)
        model.to(DEVICE)
        try:
            model.load_state_dict(st, strict=True)
            return model, None
        except Exception as e:
            # try non-strict (useful if head shape mismatch)
            try:
                model.load_state_dict(st, strict=False)
                return model, f"Loaded with strict=False (warning): {e}"
            except Exception as e2:
                return None, f"timm load failed: {e}; fallback non-strict failed: {e2}"
    except Exception as e:
        return None, f"timm create/load failed: {e}"

def try_resnet_load(state, class_names):
    """Try torchvision resnet50 fallback."""
    try:
        model = tvmodels.resnet50(pretrained=False)
        if class_names is not None:
            import torch.nn as nn
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.to(DEVICE)
        try:
            model.load_state_dict(state, strict=True)
            return model, None
        except Exception as e:
            try:
                model.load_state_dict(state, strict=False)
                return model, f"ResNet loaded with strict=False (warning): {e}"
            except Exception as e2:
                return None, f"ResNet load failed: {e}; non-strict failed: {e2}"
    except Exception as e:
        return None, f"ResNet create failed: {e}"

def try_build_model_timm(state, class_names, fallback_name="tf_efficientnet_b3"):
    """Try project helper build_model_timm (which uses timm internally)."""
    try:
        model = build_model_timm(model_name=fallback_name, num_classes=len(class_names) if class_names else 2, pretrained=False)
        model.to(DEVICE)
        try:
            model.load_state_dict(state, strict=False)
            return model, "Loaded via build_model_timm with strict=False"
        except Exception as e:
            return None, f"build_model_timm load failed: {e}"
    except Exception as e:
        return None, f"build_model_timm create failed: {e}"

def load_models_from_pattern(pattern):
    """
    Load models from a glob pattern or single file path.
    Returns (models_list, class_names, loaded_paths, diagnostics)
    """
    diagnostics = []
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0 and os.path.exists(pattern):
        paths = [pattern]
    models = []
    class_names = None
    loaded_paths = []
    if len(paths) == 0:
        diagnostics.append(f"No files matched pattern: {pattern}")
        return models, class_names, loaded_paths, diagnostics

    for p in paths:
        diagnostics.append(f"Attempting to load checkpoint: {p}")
        try:
            ck = torch.load(p, map_location="cpu")
        except Exception as e:
            diagnostics.append(f"Failed to read checkpoint file {p}: {e}")
            continue

        # extract metadata and state dict
        ck_model_name = ck.get("model_name", None)
        ck_class_names = ck.get("class_names", None) or ck.get("classes", None)
        if ck_class_names is not None:
            class_names = ck_class_names

        state = ck.get("model_state_dict", ck.get("state_dict", ck))
        if isinstance(state, dict):
            state = strip_module_prefix(state)

        model = None
        last_err = None

        # 1) Try timm if model_name present
        if ck_model_name:
            m, msg = try_timm_load(p, state, class_names or ["AI","REAL"], ck_model_name)
            if m is not None:
                models.append(m)
                loaded_paths.append(p)
                diagnostics.append(f"Loaded {p} via timm model '{ck_model_name}'" + (f" ({msg})" if msg else ""))
                continue
            else:
                diagnostics.append(f"timm attempt failed for {p}: {msg}")
                last_err = msg

        # 2) Try torchvision ResNet fallback
        m, msg = try_resnet_load(state, class_names or ["AI","REAL"])
        if m is not None:
            models.append(m)
            loaded_paths.append(p)
            diagnostics.append(f"Loaded {p} via torchvision.resnet50" + (f" ({msg})" if msg else ""))
            continue
        else:
            diagnostics.append(f"ResNet attempt failed for {p}: {msg}")
            last_err = msg

        # 3) Try build_model_timm fallback
        fallback_name = ck_model_name or "tf_efficientnet_b3"
        m, msg = try_build_model_timm(state, class_names or ["AI","REAL"], fallback_name)
        if m is not None:
            models.append(m)
            loaded_paths.append(p)
            diagnostics.append(f"Loaded {p} via build_model_timm('{fallback_name}') ({msg})")
            continue
        else:
            diagnostics.append(f"build_model_timm attempt failed for {p}: {msg}")
            last_err = msg

        diagnostics.append(f"Failed to load checkpoint {p}. Last error: {last_err}")

    return models, class_names, loaded_paths, diagnostics

# ---------------- Streamlit UI ----------------
st.markdown("Use this page to load an ensemble of saved checkpoints (e.g. model_fold*.pth). The loader will try to recreate the architecture used during training when possible.")

if st.button("Load models"):
    with st.spinner("Loading models..."):
        models, class_names, paths, diagnostics = load_models_from_pattern(MODEL_PATTERN)
        for d in diagnostics:
            st.text(d)
        if len(models) == 0:
            st.error("No models could be loaded. Check diagnostics above and ensure checkpoints are from timm or torchvision and include 'model_state_dict' with optional 'model_name' and 'class_names'.")
        else:
            # Put models in eval mode and send to device
            for m in models:
                m.eval()
            st.session_state['models'] = models
            st.session_state['class_names'] = class_names or ["AI","REAL"]
            st.success(f"Loaded {len(models)} model(s): {paths}")

uploaded = st.file_uploader("Upload image (jpg/png/webp supported)", type=['jpg','jpeg','png','webp'])
if uploaded:
    try:
        uploaded.seek(0)
    except Exception:
        pass

    try:
        img = Image.open(BytesIO(uploaded.read()))
        if img.mode in ("P", "RGBA"):
            img = img.convert("RGB")
    except UnidentifiedImageError:
        st.error("Uploaded file not recognized as an image. Try converting it to PNG/JPEG and re-upload.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()

    st.image(img, caption="Uploaded image", use_column_width=True)

    if 'models' not in st.session_state or len(st.session_state['models']) == 0:
        # Try auto-loading single best_model.pth if available
        if os.path.exists("best_model.pth"):
            st.info("No ensemble loaded — attempting to load single best_model.pth automatically.")
            models, class_names, paths, diagnostics = load_models_from_pattern("best_model.pth")
            for d in diagnostics:
                st.text(d)
            if len(models) == 0:
                st.error("Automatic load of best_model.pth failed. Please click 'Load models' and inspect diagnostics.")
                st.stop()
            else:
                for m in models:
                    m.eval()
                st.session_state['models'] = models
                st.session_state['class_names'] = class_names or ["AI","REAL"]
                st.success("Loaded best_model.pth automatically for inference.")
        else:
            st.warning("No models loaded. Click 'Load models' (or place best_model.pth in the project directory).")
            st.stop()

    # perform ensemble + TTA prediction
    models = st.session_state['models']
    class_names = st.session_state['class_names']

    try:
        probs = tta_predict_batch(models, img, input_size=input_size)
    except Exception as e:
        st.error(f"Error during model prediction: {e}\n{traceback.format_exc()}")
        st.stop()

    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    st.markdown(f"**Prediction:** {class_names[top_idx]}  \n**Confidence:** {probs[top_idx]*100:.2f}%")
    st.write("Class probabilities:")
    for i, name in enumerate(class_names):
        st.write(f"  {name}: {probs[i]*100:.2f}%")

st.markdown("---")
st.write("Notes:")
st.write("""
- This ensemble app attempts to rebuild the same architecture used during training when the checkpoint includes `model_name` and `class_names`.
- If a checkpoint was saved from a `timm` model, the loader will try `timm.create_model(model_name, ...)`. If that fails, it tries torchvision ResNet50 and a fallback `build_model_timm`.
- If you still see loading errors, inspect checkpoint contents with:
    import torch
    ck = torch.load('best_model.pth', map_location='cpu')
    print(ck.keys())
    print(list((ck.get('model_state_dict', ck)).keys())[:20])
""")
