# dashboard.py

import panel as pn
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os

pn.extension()

# Load model
model_path = 'models/best_model.pth'
class_names = ['Akara', 'Bread', 'Egusi', 'Moi Moi', 'Rice and Stew', 'Yam']
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Save image to correct folder
def save_image(image_obj, predicted_class, image_name="uploaded_image.jpg"):
    save_dir = f"data/train/{predicted_class}"
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, image_name)
    image_obj.save(image_path)

# Panel UI
image_input = pn.widgets.FileInput(accept='image/*')
output = pn.pane.Markdown("Upload an image of food 🍲")

def classify(event=None):
    if image_input.value is None:
        output.object = "⚠️ Please upload an image first."
        return
    try:
        image = Image.open(io.BytesIO(image_input.value)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]

        # Save image
        save_image(image, predicted_class)
        output.object = f"✅ Identified as **{predicted_class}**. Image saved!"
    except Exception as e:
        output.object = f"❌ Error: {str(e)}"

run_button = pn.widgets.Button(name='Classify Image', button_type='primary')
run_button.on_click(classify)

app = pn.Column(
    "# 🍽️ FlavorSnap",
    "Upload an image and get instant food classification!",
    image_input,
    run_button,
    output,
)

app.servable()
