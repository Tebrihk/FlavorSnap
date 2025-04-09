
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil

# Config
MODEL_PATH = "food_classifier.pth"
CLASS_NAMES = ["Akara", "Bread", "Egusi", "Moi Moi", "Rice and Stew", "Yam"]
SAVE_DIR = "uploaded_images"

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🍔 FlavorSnap - Food Classifier")

uploaded_file = st.file_uploader("Upload a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = CLASS_NAMES[predicted.item()]

    st.markdown(f"### ✅ Predicted Class: **{label}**")

    # Save to correct folder
    label_folder = os.path.join(SAVE_DIR, label)
    os.makedirs(label_folder, exist_ok=True)

    file_path = os.path.join(label_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Image saved to `{label_folder}`")

