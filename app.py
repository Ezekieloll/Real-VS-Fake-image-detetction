# app.py
import streamlit as st
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import torch

from model import build_model, load_checkpoint, predict_image

st.set_page_config(page_title='Fake Image Detector - Inference', layout='centered')
st.title('🖼️ Fake Image Detector — Inference (Real vs AI)')

MODEL_PATH_DEFAULT = "best_model.pth"
st.write("This app only performs inference. Train once using `train.py` (see README).")
st.write("You can optionally use `kagglehub` to download the dataset for training (train.py supports --download_kaggle).")

model_path = st.text_input('Model checkpoint path', value=MODEL_PATH_DEFAULT)
input_size = st.selectbox('Input size used at training', options=[128,160,224,288], index=2)

# Try autoload model on start
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
    if os.path.exists(model_path):
        try:
            ck = load_checkpoint(model_path, map_location='cpu')
            class_names = ck.get('class_names', ['AI','REAL'])
            model = build_model(num_classes=len(class_names), pretrained=False)
            model.load_state_dict(ck['model_state_dict'])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            st.session_state['model'] = model
            st.session_state['class_names'] = class_names
            st.session_state['model_loaded'] = True
            st.success(f"Loaded model automatically (classes: {class_names})")
        except Exception as e:
            st.warning(f"Failed to autoload model: {e}. You can train with train.py and save to {model_path}.")

if st.button("Reload model from disk"):
    if os.path.exists(model_path):
        try:
            ck = load_checkpoint(model_path, map_location='cpu')
            class_names = ck.get('class_names', ['AI','REAL'])
            model = build_model(num_classes=len(class_names), pretrained=False)
            model.load_state_dict(ck['model_state_dict'])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            st.session_state['model'] = model
            st.session_state['class_names'] = class_names
            st.session_state['model_loaded'] = True
            st.success(f"Model loaded (classes: {class_names})")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    else:
        st.error("Model file not found. Train and save model with train.py first.")

st.markdown("---")
st.write("Upload an image (jpg/png). If you get an error, the app will show diagnostics below.")

# Robust uploader handling with diagnostics + disk fallback
uploaded = st.file_uploader('Upload image (jpg/png)', type=['jpg','jpeg','png'])
if uploaded:
    st.write("Uploaded filename:", getattr(uploaded, "name", "N/A"))
    st.write("Uploaded content type:", getattr(uploaded, "type", "N/A"))

    try:
        # Ensure stream pointer at start if possible
        try:
            uploaded.seek(0)
        except Exception:
            pass

        # Read bytes once
        uploaded_bytes = uploaded.read()
        st.write("Uploaded size (bytes):", len(uploaded_bytes))

        if len(uploaded_bytes) == 0:
            st.error("Uploaded file is empty. Please try a different file.")
            st.stop()

        # Try opening from memory
        pil = None
        try:
            pil = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
            st.image(pil, caption='Uploaded image (opened from memory)', use_column_width=True)
        except UnidentifiedImageError:
            st.warning("PIL couldn't identify the image from memory. Trying to save to disk and reopen...")

            # Save to a temp file and try to open from disk (helpful for some edge cases)
            tmp_dir = os.path.join(".", "tmp_uploaded_image")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_filename = uploaded.name if getattr(uploaded, "name", None) else "tmp.jpg"
            tmp_file = os.path.join(tmp_dir, tmp_filename)
            with open(tmp_file, "wb") as f:
                f.write(uploaded_bytes)

            st.write("Saved temp file:", tmp_file, "size:", os.path.getsize(tmp_file))
            try:
                pil = Image.open(tmp_file).convert("RGB")
                st.image(pil, caption='Uploaded image (opened from disk)', use_column_width=True)
            except UnidentifiedImageError:
                st.error("PIL still could not identify the saved file. The file may be corrupted or not actually a JPEG/PNG.")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error opening saved file: {e}")
                st.stop()
        except Exception as e:
            st.error(f"Unexpected error opening image from memory: {e}")
            st.stop()

    except Exception as e:
        st.error(f"Unexpected error while processing upload: {e}")
        st.stop()

    # If we got here, 'pil' is a valid PIL.Image — continue inference
    if pil is not None:
        if not st.session_state.get('model_loaded', False):
            st.warning('Model not loaded. Train with train.py and save best_model.pth, or click Reload model from disk after placing checkpoint.')
        else:
            model = st.session_state['model']
            class_names = st.session_state['class_names']
            try:
                probs = predict_image(model, pil, input_size=input_size)
                top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                st.markdown(f'**Prediction:** {class_names[top_idx]}  \n**Confidence:** {probs[top_idx]*100:.2f}%')
                for i, cname in enumerate(class_names):
                    st.write(f'{cname}: {probs[i]*100:.2f}%')
            except Exception as e:
                st.error(f"Error during model prediction: {e}")

st.markdown("---")
st.write("Notes:")
st.write("""
- If you still get `UnidentifiedImageError` for a JPEG you know is correct, try opening it locally with a small Python script:
    from PIL import Image
    Image.open('path/to/file.jpg').convert('RGB')
  If that fails, the file is likely corrupted.
- Consider upgrading Pillow: pip install --upgrade pillow
- Temporary files are saved to ./tmp_uploaded_image when a memory-open fails.
""")
