from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import os
import base64
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from datetime import datetime
import io

app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = 'static/uploads'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# FAKE CNN MODEL (Replace with real model later)
class LungCancerModel:
    def __init__(self):
        self.weights = np.random.rand(100, 100)  # Simulated weights
    
    def predict(self, image_array):
        # Simulate CNN prediction
        features = np.mean(image_array)
        probability = 1 / (1 + np.exp(-features * 2 + np.random.randn() * 0.5))
        diagnosis = "MALIGNANT" if probability > 0.5 else "BENIGN"
        confidence = int(85 + np.random.randint(-5, 15))
        
        return {
            "diagnosis": diagnosis,
            "cancer_probability": round(float(probability), 3),
            "confidence": confidence,
            "features_detected": int(np.random.randint(5, 20))
        }

model = LungCancerModel()

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get base64 image
        data = request.get_json()
        image_b64 = data['image'].split(',')[1]
        
        # Decode image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))  # Standard CNN input size
        
        # Convert to numpy array
        image_array = np.array(image) / 255.0
        
        # AI Prediction
        result = model.predict(image_array)
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"xray_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath, 'JPEG', quality=85)
        
        return jsonify({
            "success": True,
            "result": result,
            "image_url": f"/static/uploads/{filename}"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/static/uploads/<filename>')
def uploaded_files(filename):
    return send_from_directory(os.path.join('static', 'uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
