from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import cv2
import numpy as np


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREDICT_FOLDER = 'static/predicted'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model = YOLO(r"best.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'fileUpload' not in request.files:
        return jsonify(status='error', message='No file part'), 400

    file = request.files['fileUpload']
    if file.filename == '':
        return jsonify(status='error', message='No selected file'), 400

    if file and allowed_file(file.filename):
        try:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            def enhance_image_tls_kmeans(filepath, K=3):
                img = cv2.imread(filepath)
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                stretched_hsv = np.zeros_like(hsv_img)
                for i in range(3):
                    channel = hsv_img[:, :, i]
                    low_p = np.percentile(channel, 2)
                    high_p = np.percentile(channel, 98)
                    channel_clipped = np.clip(channel, low_p, high_p)
                    stretched = ((channel_clipped - low_p) * 255 / (high_p - low_p))
                    stretched_hsv[:, :, i] = stretched.astype(np.uint8)

                Z = stretched_hsv.reshape((-1, 3)).astype(np.float32)
            
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
                centers = np.uint8(centers)
                segmented_data = centers[labels.flatten()].reshape(hsv_img.shape)
            
                return segmented_data ## applying TLS first, and then Kmeans

            #img = Image.open(filepath).convert("RGB")
            img=enhance_image_tls_kmeans(filepath)
            results = model.predict(img)
            output_img_path = os.path.join(PREDICT_FOLDER, f"predicted_{filename}")
            results[0].save(filename=output_img_path)

            return jsonify(
                status='success',
                message='Upload and prediction successful!',
                prediction_url=url_for('static', filename=f'predicted/predicted_{filename}')
            ), 200
        except Exception as e:
            return jsonify(status='error', message=f'Failed to save file: {e}'), 500

    return jsonify(status='error', message='Invalid file type'), 400

if __name__ == '__main__':
    app.run(debug=True)
