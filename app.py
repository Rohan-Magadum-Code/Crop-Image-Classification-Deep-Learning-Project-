import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import joblib

# =========================
# 🔥 FORCE CUDA → CPU FIX
# =========================
torch.serialization.default_restore_location = lambda storage, loc: storage.cpu()

# =========================
# DEVICE (CPU)
# =========================
device = torch.device("cpu")

# =========================
# LOAD MODEL FILE
# =========================
def cpu_load(path):
    return joblib.load(path)

model_data = cpu_load("crop_classifier_model.pkl")

# =========================
# BUILD MODEL (RESNET18)
# =========================
model = models.resnet18(weights=None)

num_classes = len(model_data["class_to_idx"])
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(model_data["model_state_dict"])
model.to(device)
model.eval()

# =========================
# CLASS LABELS
# =========================
idx_to_class = {v: k for k, v in model_data["class_to_idx"].items()}

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# UI
# =========================
st.title("🌾 Crop Image Classification")
st.markdown("📤 Upload an image of a crop to classify it")

uploaded_file = st.file_uploader(
    "📷 Choose an image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    st.info("🔍 Predicting...")

    # preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = idx_to_class[predicted.item()]

    st.success(f"🌱 Predicted Crop: **{predicted_class}**")