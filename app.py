import torch
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)

# Define class names in the exact order used during training
CLASS_NAMES = ['S-4T', 'S-7T', 'S-9T', 'S2T', 'S3R'] 

# Set a confidence threshold - adjust this based on testing
CONFIDENCE_THRESHOLD = 50.0  # Lowered threshold

# Temperature for softmax - values < 1 make predictions more confident
SOFTMAX_TEMPERATURE = 0.5  # Adjust this value based on testing

@app.route("/")
def home():
    return render_template("index.html")

# Load the fine-tuned model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(1280, 5)
model.load_state_dict(torch.load("Gnuts_crop_variety_classification/fine_tuned_mobilenetv2 (1).pth", map_location='cpu'), strict=True)
model.eval()

# Use the exact same transforms as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
    # Process image
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        outputs = torch.clamp(outputs, -100, 100)
        
        # Apply temperature scaling to adjust confidence
        scaled_outputs = outputs / SOFTMAX_TEMPERATURE
        probabilities = F.softmax(scaled_outputs, dim=1)
                
        if torch.isnan(probabilities).any():
            probabilities = torch.nan_to_num(probabilities, nan=0.0)
                
        predicted_idx = torch.argmax(probabilities).item()
        confidence = round(probabilities[0][predicted_idx].item() * 100, 2)
        
        # Get all class probabilities for debugging
        all_probs = {
            CLASS_NAMES[i]: round(probabilities[0][i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
        
        # Check if confidence is below threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "predicted_class": "Unknown (Not a Gnut)",
                "confidence": confidence,
                "is_gnut": False,
                "all_probabilities": all_probs  # Include for debugging
            })
        
        # If confidence is above threshold, return the prediction
        return jsonify({
            "predicted_class": CLASS_NAMES[predicted_idx],
            "confidence": confidence,
            "is_gnut": True,
            "all_probabilities": all_probs  # Include for debugging
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
