# Test script to verify model loading and prediction
import torch
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ['S2T', 'S3R', 'S-4T', 'S-7T', 'S-9T']

# Load test image
test_image = Image.open('c:/Users/KIPAALU/Desktop/imm/IMG_20230822_131924.jpg')

# Load model with CPU mapping
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, 5)
model.load_state_dict(torch.load("c:/Users/KIPAALU/Desktop/modelss/DistilledMobileNet.pth", map_location=torch.device('cpu')))
model.eval()

# Test prediction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with torch.no_grad():
    output = model(transform(test_image).unsqueeze(0))
    pred_idx = output.argmax(1).item()
    predicted_class = CLASS_NAMES[pred_idx]
    
    # Print both index and class name for verification
    print(f"Predicted index: {pred_idx}")
    print(f"Predicted class: {predicted_class}")