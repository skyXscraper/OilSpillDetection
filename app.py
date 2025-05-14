from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import os
import uuid
import urllib.request
from PIL import Image
import io
from flask import url_for, session, redirect
from flask_sqlalchemy import SQLAlchemy
import cv2
from dotenv import load_dotenv
import numpy as np
import cloudinary.uploader
import re

load_dotenv()
app = Flask(__name__)
app.secret_key = 'key1'
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    images = db.relationship('Image', backref='user', lazy=True, cascade="all, delete-orphan")

class Image(db.Model):
    __tablename__ = 'images'
    img_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id = db.Column(db.Integer, db.ForeignKey('user.id', onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    user_img_url = db.Column(db.Text, nullable=False)
    processed_img_url = db.Column(db.Text, nullable=False)

def initialize_database():
    with app.app_context():
        db.create_all()


cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

model = YOLO(r"best.pt")  

@app.route('/')
def home():
    return render_template('finalpage.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']

        # Query user using SQLAlchemy
        user = User.query.filter_by(email=email, password=password).first()
        
        if user:
            session['loggedin'] = True
            session['id'] = user.id
            session['email'] = user.email
            
            return redirect(url_for('upload'))
        else:
            msg = 'Incorrect email or password'
    return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
    
        # Check if user exists using SQLAlchemy
        existing_user = User.query.filter_by(email=email).first()
        
        if existing_user: 
            msg = 'Account already exists'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email'
        else:
            # Create new user with SQLAlchemy
            new_user = User(email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            
            msg = f'Account created successfully. Please log in with your email: {email}'
            return redirect(url_for('login')) 
    elif request.method == 'POST':
        msg = 'Please fill in all fields'
    return render_template('signup.html', msg=msg)   

def read_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
    img_array = np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_cv2

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        if f:
            res = cloudinary.uploader.upload(f.stream)
            image_url = res['secure_url']

            f.stream.seek(0)
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            def enhance_image_tls_kmeans(img, K=3):
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
                return segmented_data
            processed_img = enhance_image_tls_kmeans(img)

            results = model(processed_img)[0]
            rendered_img = results.plot()  # NumPy array with boxes

            # Upload rendered image to Cloudinary
            _, buffer = cv2.imencode('.jpg', rendered_img)                
            img_bytes = io.BytesIO(buffer)
            processed_result = cloudinary.uploader.upload(img_bytes.getvalue())
            processed_img_url = processed_result['secure_url']

            user_id = session.get('id')
            if user_id:
                # Store image using SQLAlchemy
                new_image = Image(id=user_id, user_img_url=image_url, processed_img_url=processed_img_url)
                db.session.add(new_image)
                db.session.commit()
                
                return redirect(url_for('dashboard', item=0))  # show latest image by default

            else:
                return "User not logged in", 403

    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    if 'id' not in session:
        return redirect(url_for('login'))

    user_id = session['id']
    all_images = Image.query.filter_by(id=user_id).order_by(Image.img_id.desc()).all()

    item_index = request.args.get("item", default="0")
    try:
        item_index = int(item_index)
        if item_index < 0 or item_index >= len(all_images):
            item_index = 0
    except:
        item_index = 0

    if not all_images:
        return render_template('dashboard.html', history=[], user_img_url=None, output_img_url=None, selected_index=None)

    selected = all_images[item_index]

    return render_template('dashboard.html',
                           history=all_images,
                           user_img_url=selected.user_img_url,
                           output_img_url=selected.processed_img_url,
                           selected_index=item_index)


if __name__ == "__main__":
    initialize_database() 
    app.run(debug=True)
