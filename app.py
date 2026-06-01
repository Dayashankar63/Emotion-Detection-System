# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import logging
import base64
import anthropic
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
 
face_cascade = None
try:
    haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_file)
    logger.info("Face cascade loaded")
except Exception as e:
    logger.error(f"Error: {e}")
 
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
 
        # OpenCV se face detect karo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
        # Agar OpenCV face na mile toh dummy box use karo
        if len(faces) == 0:
            h, w = frame.shape[:2]
            faces = [(int(w*0.2), int(h*0.1), int(w*0.6), int(h*0.8))]
 
        # Claude API se emotion detect karo
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data
                        }
                    },
                    {
                        "type": "text",
                        "text": "What emotion is on this face? Reply with ONE word only: angry, disgust, fear, happy, neutral, sad, or surprise."
                    }
                ]
            }]
        )
 
        emotion = message.content[0].text.strip().lower()
        if emotion not in label_emojis:
            emotion = 'neutral'
 
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'emotion': emotion,
                'emoji': label_emojis[emotion],
                'confidence': 95.0,
                'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
 
        return jsonify({'success': True, 'faces': results, 'count': len(results)})
 
    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({'error': str(e)}), 500
 
@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})
 
@app.errorhandler(404)
def not_found(e):
    return render_template('landing.html'), 404
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)