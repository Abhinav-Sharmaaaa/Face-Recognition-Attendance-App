from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from datetime import datetime
import pytz  # Import pytz for timezone handling
import cv2  # Import OpenCV
import numpy as np
import io
import logging
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import base64
import urllib.parse  # Import urllib.parse
from scipy.spatial.distance import cosine  # Import cosine from scipy

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
pymongo_logger = logging.getLogger('pymongo')
pymongo_logger.setLevel(logging.WARNING)  # Suppress pymongo debug logs

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB configuration
app.config['MONGO_URI'] = os.environ.get('MONGO_URI')
mongo = PyMongo(app)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        branch = request.form['branch']
        photo_data = request.form['photo']

        if not photo_data or not photo_data.startswith('data:image/'):
            return "Valid photo is required!", 400

        # Decode the Base64 string
        header, encoded = photo_data.split(',', 1)
        image_bytes = io.BytesIO(base64.b64decode(encoded))

        # Convert the image bytes to a numpy array
        nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return "No face detected in the image!", 400

        # Use the first detected face
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]  # Crop the face from the image

        # Resize the face image to a fixed size (e.g., 128x128)
        face_image_resized = cv2.resize(face_image, (128, 128))
        face_encoding = face_image_resized.flatten().tolist()  # Flatten the resized image

        # Save student data to MongoDB
        mongo.db.students.insert_one({
            'name': name,
            'branch': branch,
            'face_encoding': face_encoding  # Store the flattened face image
        })

        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/remove_student/<name>', methods=['POST'])
def remove_student(name):
    decoded_name = urllib.parse.unquote(name)  # Decode the name
    result = mongo.db.students.delete_one({'name': decoded_name})
    return redirect(url_for('students_view')) if result.deleted_count > 0 else ("Student not found!", 404)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance_view():
    if request.method == 'POST':
        photo_data = request.form['photo']
        print("Received photo data:", photo_data)  # Debugging line

        if not photo_data or not photo_data.startswith('data:image/'):
            return "Valid photo is required!", 400

        # Decode the Base64 string
        header, encoded = photo_data.split(',', 1)
        image_bytes = io.BytesIO(base64.b64decode(encoded))

        # Convert the image bytes to a numpy array
        nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        print(f"Detected faces: {len(faces)}")  # Debugging line

        if len(faces) == 0:
            return "No face detected in the image!", 400

        # Use the first detected face
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]  # Crop the face from the image

        # Resize the face image to a fixed size (e.g., 128x128)
        face_image_resized = cv2.resize(face_image, (128, 128))
        face_encoding = face_image_resized.flatten().tolist()  # Flatten the resized image

        # Compare with stored face encodings
        for student in mongo.db.students.find():
            stored_encoding = student['face_encoding']
            # Calculate cosine similarity
            similarity = cosine(face_encoding, stored_encoding)  # Cosine distance
            print(f"Cosine similarity to {student['name']}: {similarity}")  # Debugging line

            if similarity < 0.4:  # Adjust this threshold as needed
                # Set the timezone to Indian Standard Time (IST)
                ist = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
                mongo.db.attendance.insert_one({
                    'name': student['name'],
                    'branch': student['branch'],
                    'time': current_time
                })
                print(f"Attendance recorded for {student['name']} at {current_time}")  # Debugging line
                return redirect(url_for('index'))

        return "No matching student found!", 404

    return render_template('attendance.html')

@app.route('/students', methods=['GET'])
def students_view():
    students = mongo.db.students.find()
    return render_template('students.html', students=students)

@app.route('/present')
def present_view():
    attendance_records = list(mongo.db.attendance.find())
    print("Attendance Records:", attendance_records)
    return render_template('present.html', attendance=attendance_records)


@app.route('/reset_attendance', methods=['POST'])
def reset_attendance_route():
    mongo.db.attendance.delete_many({})
    return jsonify({"message": "Attendance has been reset."}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)