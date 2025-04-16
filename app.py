from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
from PIL import Image
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREDICT_FOLDER = 'static/predicted'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model = YOLO(r"C:/Users/muni8/OneDrive/Desktop/best.pt")

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

            img = Image.open(filepath).convert("RGB")
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
