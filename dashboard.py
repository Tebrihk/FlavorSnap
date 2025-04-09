import panel as pn
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil
from datetime import datetime

pn.extension()

# Load class names
with open("food_classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load model
model = torch.load("model.pth", map_location=torch.device('cpu'))
model.eval()

# Image transform (adjust to your model's training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload widget
upload = pn.widgets.FileInput(accept=".jpg,.png,.jpeg")

# Output widgets
output = pn.pane.Markdown("")
image_pane = pn.pane.PNG()
prediction_pane = pn.pane.Markdown("")

# Create folder if it doesn’t exist
def ensure_folder(folder):
    os.makedirs(folder, exist_ok=True)

# Handle image upload
def classify_image(event):
    if upload.value is None:
        output.object = "Please upload an image."
        return

    # Load and preprocess image
    image = Image.open(upload.file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output_probs = torch.nn.functional.softmax(model(image_tensor), dim=1)
        confidence, predicted_idx = torch.max(output_probs, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()

    # Show image and prediction
    image_pane.object = upload.value
    prediction_pane.object = f"### 🍽️ Predicted: **{predicted_class}**\n**Confidence:** {confidence:.2f}"

    # Save image to folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{predicted_class}_{timestamp}.jpg"
    save_folder = predicted_class if confidence > 0.7 else "unknown"
    ensure_folder(f"uploads/{save_folder}")
    image.save(f"uploads/{save_folder}/{filename}")

    output.object = f"📁 Image saved to: `uploads/{save_folder}/{filename}`"

# Watch for image upload
upload.param.watch(classify_image, 'value')

# Layout
dashboard = pn.Column(
    "# 🍔 FlavorSnap (Panel Edition)",
    "Upload a food image to classify and store it by category.",
    upload,
    image_pane,
    prediction_pane,
    output
)

dashboard.servable()
