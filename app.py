from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import os
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from PIL import Image
import io
import pyimgur

app = Flask(__name__)
app.secret_key = 'your_secret_key'

IMGUR_CLIENT_ID = '068bd6a3c24e96a'

FACEPP_API_KEY = 'd3exJQVyMXEhVRT-imjnp-GOPetfs6x1'
FACEPP_API_SECRET = 'pKC4ipn-73qLoaFD7fnK_t4jLjWKaM2q'
FACEPP_FACESET_TOKEN = '4ee329037a47d8ac0e5d2a85ed496126'

# Initialize storage files
STUDENTS_FILE = 'students.json'
ATTENDANCE_FILE = 'attendance.json'

# Utility functions
def load_data(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)
        return {}

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            if not isinstance(data, dict):
                data = {}
            return data
        except json.JSONDecodeError:
            with open(file_path, 'w') as f:
                json.dump({}, f)
            return {}

def save_data(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def compress_image(image_file, max_size=(1024, 1024)):
    """Compress the image to a reasonable size before uploading."""
    img = Image.open(image_file)  # image_file can be a file upload or BytesIO
    img.thumbnail(max_size)
    
    # Save the image to a BytesIO object instead of a file
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', optimize=True, quality=85)
    img_byte_arr.seek(0)  # Reset the pointer to the start of the BytesIO object
    return img_byte_arr

def upload_to_imgur(image_file):
    im = pyimgur.Imgur(IMGUR_CLIENT_ID)
    uploaded_image = im.upload_image(image_file, title="Uploaded with Flask")
    return uploaded_image.link

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
        
        # Compress the image in memory
        compressed_photo = compress_image(photo)

        # Upload to Imgur
        imgur_link = upload_to_imgur(compressed_photo)

        # Register face with Face++ API
        compressed_photo.seek(0)  # Reset the pointer to the start of the BytesIO object
        response = requests.post(
            'https://api-us.faceplusplus.com/facepp/v3/detect',
            data={'api_key': FACEPP_API_KEY, 'api_secret': FACEPP_API_SECRET},
            files={'image_file': compressed_photo}
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
        students[name] = {'branch': branch, 'photo': imgur_link, 'face_token': face_token}
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
            # Compress the image in memory
            compressed_photo = compress_image(new_photo)

            # Upload to Imgur
            imgur_link = upload_to_imgur(compressed_photo)

            # Register face with Face++ API (update face token)
            compressed_photo.seek(0)  # Reset the pointer to the start of the BytesIO object
            response = requests.post(
                'https://api-us.faceplusplus.com/facepp/v3/detect',
                data={'api_key': FACEPP_API_KEY, 'api_secret': FACEPP_API_SECRET},
                files={'image_file': compressed_photo}
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
        student_data['photo'] = imgur_link if new_photo else student_data['photo']
        students[new_name] = student_data
        if new_name != name:
            del students[name]

        save_data(STUDENTS_FILE, students)

        return redirect(url_for('index'))
    
    return render_template('edit_student.html', student=student_data)

@app.route('/remove_student/<name>', methods=['POST'])
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

    return redirect(url_for('students_view'))

@app.route('/attendance', methods=['GET', 'POST'])
def attendance_view():
    if request.method == 'POST':
        photo = request.files['photo']
        if not photo:
            return "Photo is required!", 400
        
        # Compress the image in memory
        compressed_photo = compress_image(photo)

        # Upload to Imgur
        imgur_link = upload_to_imgur(compressed_photo)

        # Detect face
        compressed_photo.seek(0)  # Reset the pointer to the start of the BytesIO object
        response = requests.post(
            'https://api-us.faceplusplus.com/facepp/v3/search',
            data={
                'api_key': FACEPP_API_KEY,
                'api_secret': FACEPP_API_SECRET,
                'faceset_token': FACEPP_FACESET_TOKEN
            },
            files={'image_file': compressed_photo}
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

@app.route('/reset_attendance', methods=['POST'])
def reset_attendance_route():
    reset_attendance()  # Call the function to reset attendance
    return jsonify({"message": "Attendance has been reset."}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable or default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)  # Listen on all network interfaces