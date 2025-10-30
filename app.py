# app.py (Final Corrected Version)# --- Part 1: Imports ---
import io
import cv2
import numpy as np
import base64
import torch
import torch.nn as nn # <-- IMPORTANT: Make sure this is imported
from flask import Flask, request, jsonify
from torchvision import models, transforms
from PIL import Image


# Flask App Initialization 
app = Flask(__name__)

# Model Loading
print("➡️  Loading model...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base model architecture
MODEL = models.densenet121()
num_features = MODEL.classifier.in_features

# **THIS IS THE FIX**: Define the classifier exactly as in your training code
MODEL.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, 2)
)

# Load the trained weights into the corrected architecture
MODEL.load_state_dict(torch.load(r"C:\Users\aryan\OneDrive\Desktop\hopescan_project\model\saved_model\best_model_checkpoint.pth", map_location=DEVICE))
MODEL.to(DEVICE)
MODEL.eval()
print(f"✅ Model ready on {DEVICE}")

# --- Part 4: Preprocessing Pipeline (No changes needed) ---
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# --- Part 5: Grad-CAM and Overlay Logic (No changes needed) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def _forward_hook(self, module, input, output):
        self.activations = output

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output).item()
            
        output[0][class_idx].backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

def overlay_heatmap(original_img_bytes, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    original_img = cv2.imdecode(np.frombuffer(original_img_bytes, np.uint8), cv2.IMREAD_COLOR)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return superimposed_img

grad_cam_generator = GradCAM(MODEL, MODEL.features.denseblock4)

# --- Part 6: The API Prediction Endpoint (No changes needed) ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        img_bytes = file.read()
        image_for_pred = Image.open(io.BytesIO(img_bytes)).convert('L')
        image_tensor = INFERENCE_TRANSFORMS(image_for_pred).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = MODEL(image_tensor)
            _, predicted_idx = torch.max(output, 1)
        
        prediction_idx = predicted_idx.item()
        prediction_label = {0: 'Benign', 1: 'Malignant'}.get(prediction_idx, 'Unknown')
        
        response_data = {'prediction': prediction_label}
        
        if prediction_idx == 1:
            heatmap = grad_cam_generator.generate_heatmap(image_tensor, class_idx=1)
            overlay_image = overlay_heatmap(img_bytes, heatmap)
            _, buffer = cv2.imencode('.jpg', overlay_image)
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data['heatmap'] = f"data:image/jpeg;base64,{heatmap_base64}"
            
        return jsonify(response_data)

# --- Part 7: Start the Server (No changes needed) ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)