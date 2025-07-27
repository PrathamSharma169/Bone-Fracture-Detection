from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import requests
import os
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
app = Flask(__name__)

# Global variables
model = None
idx_to_class = None
IMAGE_SIZE = (224, 224)

# Load model and class mapping
def load_artifacts():
    global model, idx_to_class
    try:
        print("Current working directory:", os.getcwd())
        model = tf.keras.models.load_model('model.h5')
        idx_to_class = joblib.load('idx_to_class.pkl')  # Load index-to-class mapping
        print("✅ Model and Class Mapping loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model or class mapping: {e}")
        model = None
        idx_to_class = None

# Preprocess uploaded image
def preprocess_image_from_bytes(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# Check if predicted class indicates no fracture
def is_no_fracture(fracture_type):
    """
    Check if the predicted class indicates no fracture
    Adjust these keywords based on your model's class names
    """
    no_fracture_keywords = [
        'no fracture', 'normal', 'healthy', 'no break', 
        'intact', 'unbroken', 'regular', 'fine'
    ]
    
    fracture_type_lower = fracture_type.lower()
    return any(keyword in fracture_type_lower for keyword in no_fracture_keywords)

# Gemini API Call
def get_gemini_info(fracture_type):
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        return "Gemini API key not found."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_api_key}"
    
    prompt = f"""
    Provide detailed medical information about {fracture_type} bone fracture in the following exact format:

    1. **Description**:
    What is this type of fracture and how does it occur?

    2. **Symptoms**:
    - Common signs and symptoms
    - Physical manifestations
    - Pain characteristics

    3. **Treatment Methods**:
    - Immediate first aid measures
    - Non-surgical treatment options
    - Surgical interventions if needed
    - Rehabilitation requirements

    4. **Recovery Time**:
    - Expected healing duration for different severity levels
    - Factors affecting recovery time
    - Milestones during healing

    5. **Precautions**:
    - Do's and Don'ts during recovery
    - Activities to avoid
    - Signs to watch for complications
    - When to seek immediate medical attention

    6. **Prevention**:
    - How to prevent similar injuries
    - Safety measures and recommendations
    - Lifestyle modifications

    Please use bullet points where appropriate and keep the response well-structured with clear sections.
    """

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        gemini_output = response.json()

        return gemini_output['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"Gemini API error: {e}")
        if 'response' in locals():
            print("Response content:", response.text)
        return "Could not fetch details from Gemini API."
    except KeyError:
        return "Unexpected response from Gemini API."

# Index route
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    description = None
    error_message = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error_message = 'No image file part in the request.'
        else:
            file = request.files['image']
            if file.filename == '':
                error_message = 'No selected image file.'
            elif not model or not idx_to_class:
                error_message = 'Model or Class Mapping not loaded.'
            else:
                try:
                    img_bytes = file.read()
                    print("📥 Image uploaded. Size:", len(img_bytes))
                    img_processed = preprocess_image_from_bytes(img_bytes)
                    print("📊 Processed image shape:", img_processed.shape)

                    pred = model.predict(img_processed)
                    print("🔮 Raw prediction:", pred)
                    
                    # Get prediction confidence
                    confidence = np.max(pred)
                    pred_class = np.argmax(pred, axis=1)[0]
                    fracture_type = idx_to_class.get(pred_class, "Unknown")
                    
                    print(f"✅ Predicted class: {fracture_type} (Confidence: {confidence:.4f})")
                    
                    result = fracture_type
                    
                    # Check if it's a no-fracture case
                    if is_no_fracture(fracture_type):
                        print("🎉 No fracture detected - skipping API call")
                        description = None  # No API call needed
                    else:
                        print("⚠️ Fracture detected - calling Gemini API")
                        description = get_gemini_info(fracture_type)

                except ValueError as ve:
                    error_message = f"Image processing error: {ve}"
                except Exception as e:
                    error_message = f"Unexpected error during prediction: {e}"
                    print(f"❌ Prediction error: {e}")

    return render_template('index.html', result=result, description=description, error=error_message)

# Route to test a local image (optional)
@app.route('/test', methods=['GET'])
def test_prediction():
    try:
        img_path = os.path.join("Bone Break Classification", "Fracture Dislocation", "Train", "type20i-lateral_jpg.rf.117a1f8229d0dc4d7970c07ad47c2cc1.jpg")
        if not os.path.exists(img_path):
            return jsonify({"error": f"Test image not found at {img_path}"})
        
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        
        img_processed = preprocess_image_from_bytes(img_bytes)
        pred = model.predict(img_processed)
        confidence = np.max(pred)
        pred_class = np.argmax(pred, axis=1)[0]
        fracture_type = idx_to_class.get(pred_class, "Unknown")
        
        return jsonify({
            "predicted_class": fracture_type,
            "confidence": float(confidence),
            "is_fracture": not is_no_fracture(fracture_type)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "class_mapping_loaded": idx_to_class is not None
    })

# Main
if __name__ == '__main__':
    load_artifacts()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)