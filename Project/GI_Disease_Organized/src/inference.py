
import torch
import timm
import json
import os
from torchvision import transforms
from PIL import Image

# Paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_efficientnet_b3.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_NAMES_PATH) as f:
    classes = json.load(f)

model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        pred = torch.argmax(outputs, 1).item()

    return classes[pred]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <path_to_image>")
    else:
        result = predict(sys.argv[1])
        print(f"Predicted class: {result}")
