from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from datetime import datetime
import pytz  # Import pytz for timezone handling
from PIL import Image
import io
import logging
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import base64
import urllib.parse  # Import urllib.parse
import face_recognition  # Import face_recognition library

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

        # Load the image for face recognition
        image = face_recognition.load_image_file(image_bytes)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) == 0:
            return "No face detected in the image!", 400

        # Use the first face encoding
        face_encoding = face_encodings[0]

        # Save student data to MongoDB
        mongo.db.students.insert_one({
            'name': name,
            'branch': branch,
            'face_encoding': face_encoding.tolist()  # Convert numpy array to list
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
        if not photo_data or not photo_data.startswith('data:image/'):
            return "Valid photo is required!", 400

        # Decode the Base64 string
        header, encoded = photo_data.split(',', 1)
        image_bytes = io.BytesIO(base64.b64decode(encoded))

        # Load the image for face recognition
        image = face_recognition.load_image_file(image_bytes)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) == 0:
            return "No face detected in the image!", 400

        # Compare with stored face encodings
        for student in mongo.db.students.find():
            stored_encoding = student['face_encoding']
            results = face_recognition.compare_faces([stored_encoding], face_encodings[0])

            if results[0]:
                # Set the timezone to Indian Standard Time (IST)
                ist = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
                mongo.db.attendance.insert_one({
                    'student_name': student['name'],
                    'branch': student['branch'],
                    'time': current_time
                })
                return redirect(url_for('index'))

        return "No matching face found!", 400

    return render_template('attendance.html')

@app.route('/students')
def students_view():
    students = list(mongo.db.students.find())
    return render_template('students.html', students=students)

@app.route('/present')
def present_view():
    attendance_records = list(mongo.db.attendance.find())
    return render_template('present.html', attendance=attendance_records)

@app.route('/reset_attendance', methods=['POST'])
def reset_attendance_route():
    mongo.db.attendance.delete_many({})
    return jsonify({"message": "Attendance has been reset."}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)