from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import os
from datetime import datetime
from PIL import Image
import io
import logging
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import base64
import urllib.parse  # Import urllib.parse

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

IMGUR_CLIENT_ID = '068bd6a3c24e96a'
FACEPP_API_KEY = 'd3exJQVyMXEhVRT-imjnp-GOPetfs6x1'
FACEPP_API_SECRET = 'pKC4ipn-73qLoaFD7fnK_t4jLjWKaM2q'
FACEPP_FACESET_TOKEN = '4ee329037a47d8ac0e5d2a85ed496126'

def compress_image(image_file):
    """Compress image before uploading."""
    img = Image.open(image_file)
    img.thumbnail((1024, 1024))  # Resize image

    compressed_image = io.BytesIO()
    img.save(compressed_image, format="JPEG", optimize=True, quality=85)
    compressed_image.seek(0)
    
    return compressed_image

def upload_to_imgur(image_file):
    """Uploads an image to Imgur and returns the image URL."""
    headers = {'Authorization': f'Client-ID {IMGUR_CLIENT_ID}'}
    response = requests.post(
        'https://api.imgur.com/3/image',
        headers=headers,
        files={'image': image_file}
    )

    if response.status_code == 200:
        return response.json()['data']['link']
    else:
        raise Exception(f"Imgur upload failed: {response.status_code} - {response.text}")

def add_face_to_faceset(face_token):
    """Adds a face token to the Face++ FaceSet."""
    response = requests.post(
        'https://api-us.faceplusplus.com/facepp/v3/faceset/addface',
        data={
            'api_key': FACEPP_API_KEY,
            'api_secret': FACEPP_API_SECRET,
            'faceset_token': FACEPP_FACESET_TOKEN,
            'face_tokens': face_token
        }
    )
    result = response.json()
    logging.debug(f"FaceSet add response: {result}")

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
        compressed_image = io.BytesIO(base64.b64decode(encoded))

        # Upload compressed image to Imgur
        imgur_link = upload_to_imgur(compressed_image)

        # Detect face with Face++ API
        compressed_image.seek(0)  # Reset buffer
        face_response = requests.post(
            'https://api-us.faceplusplus.com/facepp/v3/detect',
            data={'api_key': FACEPP_API_KEY, 'api_secret': FACEPP_API_SECRET},
            files={'image_file': compressed_image}
        )

        face_data = face_response.json()
        logging.debug(f"Face++ Detection Response: {face_data}")

        if 'faces' not in face_data or len(face_data['faces']) == 0:
            return "No face detected in the image!", 400

        face_token = face_data['faces'][0]['face_token']
        logging.debug(f"Registered face token: {face_token}")

        # Add face to FaceSet
        add_face_to_faceset(face_token)

        # Save student data to MongoDB
        mongo.db.students.insert_one({
            'name': name,
            'branch': branch,
            'photo': imgur_link,
            'face_token': face_token
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

        # Search for the face in the FaceSet
        search_response = requests.post(
            'https://api-us.faceplusplus.com/facepp/v3/search',
            data={
                'api_key': FACEPP_API_KEY,
                'api_secret': FACEPP_API_SECRET,
                'faceset_token': FACEPP_FACESET_TOKEN
            },
            files={'image_file': image_bytes}
        )

        result = search_response.json()
        logging.debug(f"Face++ Search Response: {result}")

        if 'results' in result and len(result['results']) > 0:
            face_token = result['results'][0]['face_token']
            confidence = result['results'][0]['confidence']
            logging.debug(f"Detected face token: {face_token}, Confidence: {confidence}")

            if confidence < 75:
                return "Face not recognized with enough confidence!", 400

            student = mongo.db.students.find_one({'face_token': face_token})
            if student:
                mongo.db.attendance.insert_one({
                    'student_name': student['name'],
                    'branch': student['branch'],
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                return redirect(url_for('index'))

            return "Student not found!", 404
        else:
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