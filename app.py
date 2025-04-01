from flask import Flask, render_template, request, redirect, url_for, jsonify, session, Response
import os
from datetime import datetime
import pytz
import cv2
import numpy as np
import io
import logging
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import base64
import urllib.parse
from scipy.spatial.distance import cosine
from werkzeug.utils import secure_filename

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
pymongo_logger = logging.getLogger('pymongo')
pymongo_logger.setLevel(logging.WARNING)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['MONGO_URI'] = os.environ.get('MONGO_URI')
mongo = PyMongo(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/live_feed')
def live_feed():
    return render_template('live_feed.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  
    while True:
        success, frame = cap.read()  
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)  

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  

                face_image = frame[y:y+h, x:x+w]
                face_image_resized = cv2.resize(face_image, (128, 128))
                face_encoding = face_image_resized.flatten().tolist()  

                # Skip attendance marking on Sundays
                if datetime.now().weekday() == 6:  
                    continue  

                today = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')
                existing_attendance = mongo.db.attendance.find_one({
                    'time': {'$regex': today}  
                })

                # If attendance is already marked today, skip marking
                if existing_attendance:
                    continue  

                best_match = None
                best_similarity = float('inf')  

                # Compare with stored face encodings
                for student in mongo.db.students.find():
                    stored_encoding = student['face_encoding']
                    similarity = cosine(face_encoding, stored_encoding)  

                    # Debugging: Print similarity scores
                    print(f"Comparing with {student['name']}: Similarity = {similarity}")

                    if similarity < best_similarity:  
                        best_similarity = similarity
                        best_match = student

                # Mark attendance if a match is found
                if best_match and best_similarity < 0.4:  
                    ist = pytz.timezone('Asia/Kolkata')
                    current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

                    # Insert attendance record
                    mongo.db.attendance.insert_one({
                        'name': best_match['name'],
                        'branch': best_match['branch'],
                        'time': current_time
                    })
                    print(f"Attendance recorded for {best_match['name']} at {current_time}")

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/student_index')
def student_index():
    if session.get('user_type') == 'student':
        return render_template('student_index.html')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        branch = request.form['branch']
        photo_data = request.form['photo']  
        uploaded_file = request.files.get('uploadPhoto')  

        
        if not photo_data and not uploaded_file:
            return "A photo is required!", 400

        existing_student = mongo.db.students.find_one({'name': name})
        if existing_student:
            return "Student already registered!", 400

        image = None

        if photo_data and photo_data.startswith('data:image/'):
            
            header, encoded = photo_data.split(',', 1)
            image_bytes = io.BytesIO(base64.b64decode(encoded))
            nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif uploaded_file:
            
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            image = cv2.imread(file_path)

        if image is None:
            return "No valid image provided!", 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return "No face detected in the image!", 400

        
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]  

        
        face_image_resized = cv2.resize(face_image, (128, 128))
        face_encoding = face_image_resized.flatten().tolist()  

        
        mongo.db.students.insert_one({
            'name': name,
            'branch': branch,
            'face_encoding': face_encoding  
        })

        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/remove_student/<name>', methods=['POST'])
def remove_student(name):
    decoded_name = urllib.parse.unquote(name)  
    result = mongo.db.students.delete_one({'name': decoded_name})
    return redirect(url_for('students_view')) if result.deleted_count > 0 else ("Student not found!", 404)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance_view():
    if request.method == 'POST':
        photo_data = request.form['photo']  
        uploaded_file = request.files.get('uploadPhoto')  

        
        if not photo_data and not uploaded_file:
            return "A photo is required!", 400

        image = None

        if photo_data and photo_data.startswith('data:image/'):
            
            header, encoded = photo_data.split(',', 1)
            image_bytes = io.BytesIO(base64.b64decode(encoded))
            nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif uploaded_file:
            
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            image = cv2.imread(file_path)

        if image is None:
            return "No valid image provided!", 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return "No face detected in the image!", 400

        
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]  

        
        face_image_resized = cv2.resize(face_image, (128, 128))
        face_encoding = face_image_resized.flatten().tolist()  

        
        if datetime.now().weekday() == 6:  
            return render_template('holiday.html')  

        
        today = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')
        existing_attendance = mongo.db.attendance.find_one({
            'time': {'$regex': today}  
        })

        if existing_attendance:
            return render_template('already_marked.html')  

        best_match = None
        best_similarity = float('inf')  
        student_name = None  

        
        for student in mongo.db.students.find():
            stored_encoding = student['face_encoding']
            similarity = cosine(face_encoding, stored_encoding)  
            print(f"Cosine similarity to {student['name']}: {similarity}")  

            if similarity < best_similarity:  
                best_similarity = similarity
                best_match = student
                student_name = student['name']  

        if best_match and best_similarity < 0.4:  
            
            ist = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

            
            mongo.db.attendance.insert_one({
                'name': student_name,  
                'branch': best_match['branch'],
                'time': current_time
            })
            print(f"Attendance recorded for {student_name} at {current_time}")  
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

@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'photo' not in request.files:
            return "No file part", 400
        file = request.files['photo']
        if file.filename == '':
            return "No selected file", 400
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('index'))

    return render_template('upload_photo.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_type = request.form.get('user_type')
        if user_type == 'admin':
            admin_password = request.form.get('admin_password')
            if admin_password == '0000':  
                session['user_type'] = user_type
                print(f"User    type set in session: {session['user_type']}")  
                return redirect(url_for('index'))
            else:
                return "Invalid admin password!", 403  
        elif user_type == 'student':
            session['user_type'] = user_type
            print(f"User    type set in session: {session['user_type']}")  
            return redirect(url_for('student_index'))  
        return "Invalid user type!", 400
    return render_template('login.html')  

@app.route('/logout')
def logout():
    session.pop('user_type', None)
    return redirect(url_for('login'))

@app.before_request
def restrict_access():
    allowed_routes = ['login', 'static', 'logout']
    if request.endpoint in allowed_routes:
        return None

    if 'user_type' not in session:
        return redirect(url_for('login'))

    
    if session.get('user_type') == 'admin':
        return None

    
    if session.get('user_type') == 'student':
        if request.endpoint in ['attendance_view', 'student_index']:
            return None

    return redirect(url_for('login'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    app.run(host='0.0.0.0', port=port, debug=True)