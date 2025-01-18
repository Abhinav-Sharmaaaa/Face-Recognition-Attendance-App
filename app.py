from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import os
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.secret_key = 'your_secret_key'

FACEPP_API_KEY = 'd3exJQVyMXEhVRT-imjnp-GOPetfs6x1'
FACEPP_API_SECRET = 'pKC4ipn-73qLoaFD7fnK_t4jLjWKaM2q'
FACEPP_FACESET_TOKEN = '4ee329037a47d8ac0e5d2a85ed496126'

# Initialize storage files
STUDENTS_FILE = 'students.json'
ATTENDANCE_FILE = 'attendance.json'

# Utility functions
def load_data(file_path):
    if not os.path.exists(file_path):
        # Create an empty dictionary in the file if it doesn't exist
        with open(file_path, 'w') as f:
            json.dump({}, f)
        return {}

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            # Ensure the data is a dictionary
            if not isinstance(data, dict):
                data = {}
            return data
        except json.JSONDecodeError:
            # Reset to an empty dictionary if JSON is invalid
            with open(file_path, 'w') as f:
                json.dump({}, f)
            return {}

def save_data(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def compress_image(image_path, max_size=(1024, 1024)):
    """Compress the image to a reasonable size before saving."""
    img = Image.open(image_path)
    img.thumbnail(max_size)
    
    # Ensure the directory exists before saving the image
    compressed_image_dir = os.path.dirname(image_path)
    if not os.path.exists(compressed_image_dir):
        os.makedirs(compressed_image_dir)
    
    # Construct the path for the compressed image
    base_filename, ext = os.path.splitext(image_path)
    compressed_image_path = base_filename + '_compressed' + ext
    
    img.save(compressed_image_path, optimize=True, quality=85)
    return compressed_image_path

# Initialize files
students = load_data(STUDENTS_FILE)
attendance = load_data(ATTENDANCE_FILE)

# Initialize the scheduler
scheduler = BackgroundScheduler()

def reset_attendance():
    """This function resets the attendance data at midnight."""
    global attendance
    attendance = {}  # Reset the attendance dictionary
    save_data(ATTENDANCE_FILE, attendance)
    print("Attendance reset at midnight")

# Schedule the reset task to run every day at midnight
scheduler.add_job(func=reset_attendance, trigger='cron', hour=0, minute=0)

# Start the scheduler
scheduler.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        branch = request.form['branch']
        photo = request.files['photo']
        
        if not photo:
            return "Photo is required!", 400
        
        # Save photo locally
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
        photo.save(photo_path)

        # Compress the image before uploading
        compressed_photo_path = compress_image(photo_path)

        # Register face with Face++ API
        with open(compressed_photo_path, 'rb') as image_file:
            response = requests.post(
                'https://api-us.faceplusplus.com/facepp/v3/detect',
                data={'api_key': FACEPP_API_KEY, 'api_secret': FACEPP_API_SECRET},
                files={'image_file': image_file}
            )
        
        face_data = response.json()
        if 'faces' not in face_data or len(face_data['faces']) == 0:
            return "No face detected in the image!", 400
        
        face_token = face_data['faces'][0]['face_token']

        # Add face to FaceSet
        requests.post(
            'https://api-us.faceplusplus.com/facepp/v3/faceset/addface',
            data={
                'api_key': FACEPP_API_KEY,
                'api_secret': FACEPP_API_SECRET,
                'faceset_token': FACEPP_FACESET_TOKEN,
                'face_tokens': face_token
            }
        )

        # Save student data
        students[name] = {'branch': branch, 'photo': photo.filename, 'face_token': face_token}
        save_data(STUDENTS_FILE, students)

        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/edit_student/<name>', methods=['GET', 'POST'])
def edit_student(name):
    if name not in students:
        return "Student not found!", 404

    student_data = students[name]
    
    if request.method == 'POST':
        new_name = request.form['name']
        new_branch = request.form['branch']
        new_photo = request.files['photo']
        
        if new_photo:
            # Save the new photo
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], new_photo.filename)
            new_photo.save(photo_path)

            # Compress the image before uploading
            compressed_photo_path = compress_image(photo_path)

            # Register face with Face++ API (update face token)
            with open(compressed_photo_path, 'rb') as image_file:
                response = requests.post(
                    'https://api-us.faceplusplus.com/facepp/v3/detect',
                    data={'api_key': FACEPP_API_KEY, 'api_secret': FACEPP_API_SECRET},
                    files={'image_file': image_file}
                )
            
            face_data = response.json()
            if 'faces' not in face_data or len(face_data['faces']) == 0:
                return "No face detected in the image!", 400
            
            face_token = face_data['faces'][0]['face_token']

            # Remove old face from FaceSet and add new face
            requests.post(
                'https://api-us.faceplusplus.com/facepp/v3/faceset/removeface',
                data={
                    'api_key': FACEPP_API_KEY,
                    'api_secret': FACEPP_API_SECRET,
                    'faceset_token': FACEPP_FACESET_TOKEN,
                    'face_tokens': student_data['face_token']
                }
            )
            requests.post(
                'https://api-us.faceplusplus.com/facepp/v3/faceset/addface',
                data={
                    'api_key': FACEPP_API_KEY,
                    'api_secret': FACEPP_API_SECRET,
                    'faceset_token': FACEPP_FACESET_TOKEN,
                    'face_tokens': face_token
                }
            )

            student_data['face_token'] = face_token

        # Update the student information
        student_data['branch'] = new_branch
        student_data['photo'] = new_photo.filename if new_photo else student_data['photo']
        students[new_name] = student_data
        if new_name != name:
            del students[name]

        save_data(STUDENTS_FILE, students)

        return redirect(url_for('index'))
    
    return render_template('edit_student.html', student=student_data)

@app.route('/remove_student/<name>', methods=['GET'])
def remove_student(name):
    if name not in students:
        return "Student not found!", 404

    student_data = students.pop(name)
    
    # Remove the face from Face++ FaceSet
    requests.post(
        'https://api-us.faceplusplus.com/facepp/v3/faceset/removeface',
        data={
            'api_key': FACEPP_API_KEY,
            'api_secret': FACEPP_API_SECRET,
            'faceset_token': FACEPP_FACESET_TOKEN,
            'face_tokens': student_data['face_token']
        }
    )

    save_data(STUDENTS_FILE, students)

    return redirect(url_for('index'))

@app.route('/attendance', methods=['GET', 'POST'])
def attendance_view():
    if request.method == 'POST':
        photo = request.files['photo']
        if not photo:
            return "Photo is required!", 400
        
        # Save photo locally
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
        photo.save(photo_path)

        # Compress the image before uploading
        compressed_photo_path = compress_image(photo_path)

        # Detect face
        with open(compressed_photo_path, 'rb') as image_file:
            response = requests.post(
                'https://api-us.faceplusplus.com/facepp/v3/search',
                data={
                    'api_key': FACEPP_API_KEY,
                    'api_secret': FACEPP_API_SECRET,
                    'faceset_token': FACEPP_FACESET_TOKEN
                },
                files={'image_file': image_file}
            )
        
        result = response.json()
        if 'results' not in result or len(result['results']) == 0:
            return "No matching face found!", 400
        
        face_token = result['results'][0]['face_token']
        for name, data in students.items():
            if data['face_token'] == face_token:
                attendance[name] = {'branch': data['branch'], 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                save_data(ATTENDANCE_FILE, attendance)
                return redirect(url_for('index'))
        
        return "Student not found!", 404
    return render_template('attendance.html')

@app.route('/students')
def students_view():
    return render_template('students.html', students=students)

@app.route('/present')
def present_view():
    return render_template('present.html', attendance=attendance)

if __name__ == '__main__':
    app.run(debug=True)
