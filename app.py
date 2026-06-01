from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import logging
import base64
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
# Load model - tensorflow optional
model = None
try:
    import tensorflow as tf
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = tf.keras.models.load_model(os.path.join(base_dir, "emotiondetector.h5"))
    logger.info("Model loaded successfully")
except ImportError:
    logger.warning("TensorFlow not installed - model unavailable")
except Exception as e:
    logger.error(f"Error loading model: {e}")
 
# Load face cascade
face_cascade = None
try:
    haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_file)
    logger.info("Face cascade loaded")
except Exception as e:
    logger.error(f"Error loading face cascade: {e}")
 
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
label_emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}
 
@app.route('/')
def index():
    return render_template('landing.html')
 
@app.route('/detect')
def detect():
    return render_template('detect.html')
 
@app.route('/api/predict', methods=['POST'])
def predict():
    """Browser se base64 image leke emotion detect karo"""
    if model is None:
        return jsonify({'error': 'Model not loaded - TensorFlow unavailable'}), 503
 
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
 
        img_data = data['image']
        if ',' in img_data:
            img_data = img_data.split(',')[1]
 
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
 
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
        results = []
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            feature = np.array(face_img).reshape(1, 48, 48, 1) / 255.0
            pred = model.predict(feature, verbose=0)
            emotion = labels[pred.argmax()]
            confidence = float(pred.max())
            results.append({
                'emotion': emotion,
                'emoji': label_emojis[emotion],
                'confidence': round(confidence * 100, 1),
                'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
 
        return jsonify({'success': True, 'faces': results, 'count': len(results)})
 
    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({'error': str(e)}), 500
 
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})
 
@app.errorhandler(404)
def not_found(e):
    return render_template('landing.html'), 404
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)