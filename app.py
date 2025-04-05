from flask import Flask, render_template, request, redirect, url_for, jsonify, session, Response, flash
import os
from datetime import datetime
import json
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

logging.basicConfig(level=logging.INFO)
pymongo_logger = logging.getLogger('pymongo')
pymongo_logger.setLevel(logging.WARNING)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", 'default_fallback_secret_key')

app.config['MONGO_URI'] = os.environ.get('MONGO_URI')
if not app.config['MONGO_URI']:
    logging.error("MONGO_URI not set in environment variables.")
    exit("MongoDB URI is required.")
mongo = PyMongo(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

yunet_model_path = 'yunet_s_640_640.onnx'
sface_model_path = 'face_recognition_sface_2021dec.onnx' 


if not os.path.exists(yunet_model_path):
    logging.error(f"YuNet model file not found at: {yunet_model_path}")
    exit("YuNet model file missing.")
if not os.path.exists(sface_model_path): 
    logging.error(f"SFace model file not found at: {sface_model_path}")
    exit("SFace model file missing.")

detector_input_size = (320, 320) 
detector_score_threshold = 0.9
detector_nms_threshold = 0.3
detector_top_k = 5000
face_detector = None
try:
    face_detector = cv2.FaceDetectorYN.create(
        model=yunet_model_path,
        config="",
        input_size=detector_input_size,
        score_threshold=detector_score_threshold,
        nms_threshold=detector_nms_threshold,
        top_k=detector_top_k
    )
    logging.info("YuNet face detector loaded successfully.")
except cv2.error as e:
    logging.error(f"Error loading YuNet model: {e}")
    exit("Failed to load face detector.")
except AttributeError:
     logging.error("cv2.FaceDetectorYN not found. Check OpenCV installation (needs version >= 4.5.4).")
     exit("OpenCV FaceDetectorYN unavailable.")


face_recognizer = None
try:
    face_recognizer = cv2.FaceRecognizerSF.create(
        model=sface_model_path,
        config="",
        backend_id=0, 
        target_id=0 
    )
    logging.info("SFace face recognizer loaded successfully.")
except cv2.error as e:
    logging.error(f"Error loading SFace model: {e}")
    exit("Failed to load face recognizer.")
except AttributeError:
     logging.error("cv2.FaceRecognizerSF not found. Check OpenCV installation (needs version >= 4.5.4).")
     exit("OpenCV FaceRecognizerSF unavailable.")


known_face_data = {}
known_face_embeddings_list = [] 
known_face_names = []





RECOGNITION_THRESHOLD = 0.36 

def load_known_faces():
    global known_face_data, known_face_embeddings_list, known_face_names
    logging.info("Loading known faces from database...")
    known_face_data = {}
    temp_embeddings_list = []
    temp_names_list = []
    try:
        
        students_cursor = mongo.db.students.find({}, {"name": 1, "branch": 1, "face_embedding": 1}) 
        count = 0
        for student in students_cursor:
            
            if 'face_embedding' in student and student['face_embedding'] and 'name' in student:
                try:
                    
                    embedding_np = np.array(student['face_embedding']).astype(np.float32)
                    
                    if embedding_np.shape == (128,): 
                        known_face_data[student['name']] = {
                            'embedding': embedding_np, 
                            'branch': student.get('branch', 'N/A')
                        }
                        temp_embeddings_list.append(embedding_np)
                        temp_names_list.append(student['name'])
                        count += 1
                    else:
                        logging.warning(f"Invalid embedding shape {embedding_np.shape} for student {student.get('name', 'N/A')}. Expected (128,). Skipping.")

                except Exception as e:
                    logging.warning(f"Could not process embedding for student {student.get('name', 'N/A')}: {e}")
            else:
                logging.warning(f"Student data incomplete or missing face_embedding: {student.get('_id')}")

        known_face_embeddings_list = temp_embeddings_list 
        known_face_names = temp_names_list
        logging.info(f"Loaded {count} known faces (embeddings) into memory cache.")

    except Exception as e:
        logging.error(f"Error loading known faces from database: {e}")


def detect_and_encode_face(image_np):
    """
    Detects the most prominent face using YuNet and generates its SFace embedding.

    Args:
        image_np: NumPy array of the input image (BGR format).

    Returns:
        tuple: (face_embedding, bounding_box) or (None, None) if no face detected/processed.
               face_embedding is a NumPy array (128,) or None.
               bounding_box is (x, y, w, h) or None.
    """
    if image_np is None or image_np.size == 0:
        logging.warning("detect_and_encode_face received an empty image.")
        return None, None
    if face_detector is None or face_recognizer is None:
         logging.error("Detector or Recognizer not initialized.")
         return None, None

    height, width, _ = image_np.shape
    
    face_detector.setInputSize((width, height))

    try:
        
        status, faces = face_detector.detect(image_np)
    except cv2.error as e:
        logging.error(f"Error during face detection: {e}")
        return None, None

    if faces is None or len(faces) == 0:
        
        return None, None

    
    
    best_face_index = np.argmax(faces[:, -1])
    face_data = faces[best_face_index]
    box = face_data[0:4].astype(np.int32)
    (x, y, w, h) = box

    
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    if w <= 0 or h <= 0:
        logging.warning(f"Invalid bounding box after boundary check: x={x}, y={y}, w={w}, h={h}")
        return None, (x,y,w,h) 

    try:
        
        
        aligned_face = face_recognizer.alignCrop(image_np, face_data)

        if aligned_face is None or aligned_face.size == 0:
            logging.warning("Face alignment/cropping failed.")
            return None, (x,y,w,h) 

        
        face_embedding = face_recognizer.feature(aligned_face)
        
        face_embedding_flat = face_embedding.flatten()

        
        return face_embedding_flat, (x, y, w, h)

    except cv2.error as e:
        logging.error(f"Error during face alignment or feature extraction: {e}")
        return None, (x,y,w,h) 
    except Exception as e:
        logging.error(f"Unexpected error during encoding: {e}")
        return None, (x,y,w,h)


load_known_faces()

@app.route('/refresh_faces', methods=['POST'])
def refresh_faces_route():
     if session.get('user_type') != 'admin':
         return jsonify({"message": "Unauthorized"}), 403
     try:
         load_known_faces() 
         return jsonify({"message": f"Face cache refreshed. Loaded {len(known_face_data)} faces."}), 200
     except Exception as e:
         logging.error(f"Error during manual face cache refresh: {e}")
         return jsonify({"message": "Error refreshing face cache."}), 500

@app.route('/')
def index():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/live_feed')
def live_feed():
    
    if session.get('user_type') == 'admin':
         return render_template('live_feed.html')
    elif session.get('user_type') == 'student':
         return redirect(url_for('student_index'))
    else:
         return redirect(url_for('login'))


def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera 0")
        return

    frame_count = 0
    process_every_n_frames = 3
    last_known_faces = {}
    session_seen_known_faces = {} # Track known faces seen in this generator session
    india_tz = pytz.timezone('Asia/Kolkata')
    seen_log_collection = mongo.db.seen_log # Collection for logging seen faces

    while True:
        success, frame = cap.read()
        if not success:
            logging.warning("Failed to read frame from camera.")
            break

        
        processing_frame = frame.copy()
        frame = cv2.flip(frame, 1) 
        processing_frame = cv2.flip(processing_frame, 1) 
        frame_count += 1

        current_detected_faces = {} 

        
        if frame_count % process_every_n_frames == 0:
            face_embedding, box = detect_and_encode_face(processing_frame) 
            status_text = ""
            status_color = (0, 0, 0) 

            if box: 
                current_time_india = datetime.now(india_tz)

                
                if current_time_india.weekday() == 6: 
                    status_text = "Sunday - No Attendance"
                    status_color = (255, 255, 0) 
                    current_detected_faces[0] = {'box': box, 'status': status_text, 'color': status_color}
                
                elif face_embedding is not None:
                    best_match_name = None
                    best_similarity = float('inf') 

                    
                    for i, stored_embedding in enumerate(known_face_embeddings_list):
                        try:
                            
                            if face_embedding.shape != stored_embedding.shape:
                                logging.warning(f"Shape mismatch: current {face_embedding.shape}, stored {stored_embedding.shape}")
                                continue

                            similarity = cosine(face_embedding, stored_embedding)

                            if similarity < best_similarity:
                                best_similarity = similarity
                                best_match_name = known_face_names[i]

                        except Exception as e:
                            logging.error(f"Error comparing embedding {i} for {known_face_names[i]}: {e}")
                            continue 

                    student_name = None
                    
                    if best_match_name and best_similarity < RECOGNITION_THRESHOLD:
                        student_name = best_match_name
                        logging.info(f"Live Feed Match: {student_name} (Cosine Dist: {best_similarity:.4f})") 

                        
                        today_str = current_time_india.strftime('%Y-%m-%d')
                        existing_attendance_student = mongo.db.attendance.find_one({
                            'name': student_name,
                            'time': {'$regex': f'^{today_str}'}
                        })

                        if not existing_attendance_student:
                            current_time_str = current_time_india.strftime('%Y-%m-%d %H:%M:%S')
                            student_branch = known_face_data.get(student_name, {}).get('branch', 'N/A')
                            try:
                                mongo.db.attendance.insert_one({
                                    'name': student_name,
                                    'branch': student_branch,
                                    'time': current_time_str
                                })
                                logging.info(f"Attendance recorded via live feed for {student_name} at {current_time_str}")
                                status_text = f"Marked: {student_name}"
                                status_color = (0, 255, 0) 
                            except Exception as e:
                                logging.error(f"DB error inserting attendance for {student_name}: {e}")
                                status_text = "DB Error Marking"
                                status_color = (0, 165, 255) 
                        else:
                            
                            status_text = f"Already Marked: {student_name}"
                            status_color = (0, 255, 255) 
                    
                    elif best_match_name:
                        
                        status_text = f"Low Match: {best_match_name} ({best_similarity:.2f})" 
                        status_color = (0, 0, 255) 
                    else:
                        
                        status_text = "Unknown Face"
                        status_color = (255, 0, 0) 

                    current_detected_faces[0] = {'box': box, 'status': status_text, 'color': status_color}

                else: 
                     status_text = "Processing Error"
                     status_color = (255, 0, 255) 
                     current_detected_faces[0] = {'box': box, 'status': status_text, 'color': status_color}

                # --- Refined Logging Logic ---
                if box: # Only proceed if a face box was detected
                    if student_name: # Known face detected
                        # Removed session check to log every sighting
                        # session_seen_known_faces[student_name] = current_time_india # Mark as seen in this session
                        log_entry = {
                            'type': 'known_sighting',
                                'name': student_name,
                                'timestamp': current_time_india,
                                'status_at_log': status_text # Log the status when first seen
                            }
                        try: # <-- Corrected indentation
                            seen_log_collection.insert_one(log_entry)
                            logging.info(f"Logged sighting of known face: {student_name}") # Changed log message slightly
                        except Exception as db_err:
                            logging.error(f"Error inserting known sighting log for {student_name}: {db_err}")
                        # No 'else' needed here, log every time if known

                    elif status_text == "Unknown Face": # Unknown face detected
                        # Log unknown faces every time they appear
                        log_entry = {
                            'type': 'unknown_sighting',
                            'timestamp': current_time_india,
                            'status_at_log': status_text
                        }
                        # Add cropped face image for unknown faces
                        try:
                            (x, y, w, h) = box
                            if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= processing_frame.shape[1] and y + h <= processing_frame.shape[0]:
                                cropped_face = processing_frame[y:y+h, x:x+w]
                                if cropped_face.size > 0:
                                    _, buffer = cv2.imencode('.jpg', cropped_face)
                                    log_entry['face_image_base64'] = base64.b64encode(buffer).decode('utf-8')
                                    try:
                                        seen_log_collection.insert_one(log_entry)
                                        logging.info(f"Logged unknown face sighting with image.")
                                    except Exception as db_err:
                                        logging.error(f"Error inserting unknown sighting log: {db_err}")
                                else:
                                    logging.warning("Cropped unknown face is empty, skipping log.")
                            else:
                                logging.warning(f"Invalid box coordinates for cropping unknown face: {box}, frame shape: {processing_frame.shape}")
                        except Exception as crop_err:
                                logging.error(f"Error cropping/encoding unknown face for seen log: {crop_err}")
                # --- End Logging Logic --- # Removed duplicate logging block

            # Update last known faces for display smoothing
            last_known_faces = current_detected_faces if box else {} # Keep this for smoother display

            display_faces = current_detected_faces

        else:
            
            display_faces = last_known_faces

        
        for face_id, face_info in display_faces.items():
             (x, y, w, h) = face_info['box']
             cv2.rectangle(frame, (x, y), (x + w, y + h), face_info['color'], 2)
             
             cv2.putText(frame, face_info['status'], (x, y - 10 if y > 10 else y + 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_info['color'], 2)

        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.warning("Failed to encode frame to JPEG.")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    logging.info("Camera released.")


@app.route('/video_feed')
def video_feed():
     
     if session.get('user_type') != 'admin':
         logging.warning("Unauthorized attempt to access video_feed endpoint.")
         
         return Response("Unauthorized", status=403)
         

     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/student_index')
def student_index():
    if session.get('user_type') == 'student':
        return render_template('student_index.html')
    
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_type') != 'admin':
       return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form.get('name')
        branch = request.form.get('branch')
        photo_data = request.form.get('photo') 
        uploaded_file = request.files.get('uploadPhoto') 

        if not name or not branch:
             return "Name and Branch are required!", 400

        image = None
        image_source = "unknown"
        file_path = None 

        
        if photo_data and photo_data.startswith('data:image/'):
            try:
                
                header, encoded = photo_data.split(',', 1)
                image_bytes = io.BytesIO(base64.b64decode(encoded))
                nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image_source = "webcam"
                logging.info("Processing image from webcam data.")
            except Exception as e:
                logging.error(f"Error decoding base64 image: {e}")
                return "Invalid photo data from webcam.", 400
        elif uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                uploaded_file.save(file_path)
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"cv2.imread returned None for {filename}. Check file format/integrity.")
                image_source = f"upload ({filename})"
                logging.info(f"Processing image from uploaded file: {filename}")
            except Exception as e:
                logging.error(f"Error saving or reading uploaded file '{filename}': {e}")
                
                if file_path and os.path.exists(file_path):
                    try: os.remove(file_path)
                    except OSError as rm_err: logging.error(f"Error removing temporary file {file_path}: {rm_err}")
                return f"Could not process uploaded file: {e}", 400
        else:
            return "No photo provided (either webcam capture or file upload is required).", 400
        
        if image is None:
            
            logging.error("Image data is None after loading attempts.")
            if file_path and os.path.exists(file_path): 
                 try: os.remove(file_path)
                 except OSError as rm_err: logging.error(f"Error removing file {file_path} after load fail: {rm_err}")
            return "Failed to load image data.", 500

        existing_student = mongo.db.students.find_one({'name': name})
        if existing_student:
            logging.warning(f"Registration attempt for existing student: {name}")
            if file_path and os.path.exists(file_path): 
                try: os.remove(file_path)
                except OSError as e: logging.error(f"Error removing upload file {file_path} for existing student: {e}")
            return f"Student '{name}' is already registered!", 400
        
        logging.info(f"Processing registration for {name} from {image_source}.")
        face_embedding, box = detect_and_encode_face(image) 
        
        if face_embedding is None:
            logging.warning(f"No face detected or embedding failed for registration image of {name}.")
            if file_path and os.path.exists(file_path): 
                try: os.remove(file_path)
                except OSError as e: logging.error(f"Error removing upload file {file_path} after no face detected: {e}")
            
            if box:
                 message = "Face detected, but could not process it for registration. Try a clearer image or different angle."
            else:
                 message = "No face detected in the provided image. Please ensure the face is clear and well-lit."
            return message, 400
        
        try:
            india_tz = pytz.timezone('Asia/Kolkata')
            
            embedding_list = face_embedding.tolist()
            mongo.db.students.insert_one({
                'name': name,
                'branch': branch,
                'face_embedding': embedding_list, 
                'registered_at': datetime.now(india_tz)
            })
            logging.info(f"Successfully registered student: {name}")
            load_known_faces() 
        except Exception as e:
            logging.error(f"Database error during registration for {name}: {e}")
            if file_path and os.path.exists(file_path): 
                 try: os.remove(file_path)
                 except OSError as rm_err: logging.error(f"Error removing upload file {file_path} after DB error: {rm_err}")
            return "Database error occurred during registration.", 500
        
        if file_path and os.path.exists(file_path):
             try: os.remove(file_path)
             except OSError as e: logging.error(f"Error removing upload file {file_path} after successful registration: {e}")

        return redirect(url_for('students_view')) 
    
    return render_template('register.html')

@app.route('/remove_student/<name>', methods=['POST'])
def remove_student(name):
    if session.get('user_type') != 'admin':
        return "Unauthorized", 403

    try:
        
        decoded_name = urllib.parse.unquote(name)
        logging.info(f"Attempting to remove student: {decoded_name}")
        
        result = mongo.db.students.delete_one({'name': decoded_name})

        if result.deleted_count > 0:
            logging.info(f"Successfully removed student: {decoded_name}")
            load_known_faces() 
                        
            return redirect(url_for('students_view'))
        else:
            logging.warning(f"Student not found for removal: {decoded_name}")
            
            return f"Student '{decoded_name}' not found!", 404 

    except Exception as e:
         logging.error(f"Error removing student {name}: {e}")
         
         return "An error occurred while trying to remove the student.", 500

@app.route('/attendance', methods=['GET', 'POST'])
def attendance_view():
    user_type = session.get('user_type')
    india_tz = pytz.timezone('Asia/Kolkata')

    if request.method == 'POST':
        if user_type != 'student':
            return "Unauthorized: Only students can mark attendance here.", 403

        photo_data = request.form.get('photo')
        uploaded_file = request.files.get('uploadPhoto')
        
        if datetime.now(india_tz).weekday() == 6: 
            logging.info("Attendance attempt on Sunday.")
            
            return render_template('attendance_result.html', success=False, message="Today is Sunday, attendance is not recorded.")
                
        image = None
        image_source = "unknown"
        file_path = None
        
        if photo_data and photo_data.startswith('data:image/'):
            try:
                header, encoded = photo_data.split(',', 1)
                image_bytes = io.BytesIO(base64.b64decode(encoded))
                nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image_source = "webcam"
            except Exception as e:
                logging.error(f"Error decoding base64 image for attendance: {e}")
                return render_template('attendance_result.html', success=False, message="Invalid photo data from webcam.")
        elif uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(f"attendance_{uploaded_file.filename}") 
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                uploaded_file.save(file_path)
                image = cv2.imread(file_path)
                if image is None: raise ValueError("cv2.imread returned None")
                image_source = f"upload ({uploaded_file.filename})"
                
                if file_path and os.path.exists(file_path):
                    try: os.remove(file_path)
                    except OSError as rm_err: logging.error(f"Error removing temp attendance file {file_path}: {rm_err}")
            except Exception as e:
                logging.error(f"Error saving/reading uploaded file for attendance '{filename}': {e}")
                if file_path and os.path.exists(file_path): 
                     try: os.remove(file_path)
                     except OSError as rm_err: logging.error(f"Error removing temp attendance file {file_path} on error: {rm_err}")
                return render_template('attendance_result.html', success=False, message=f"Could not process uploaded file: {e}")
        else:
             return render_template('attendance_result.html', success=False, message="No photo provided for attendance.")

        if image is None:
             return render_template('attendance_result.html', success=False, message="Failed to load image data for attendance.")

        logging.info(f"Processing attendance attempt from {image_source}.")
        face_embedding, box = detect_and_encode_face(image)

        if face_embedding is None:
            logging.warning("No face detected or embedding failed for attendance image.")
            message = "No face detected in the image." if not box else "Face detected, but could not be processed. Try again."
            return render_template('attendance_result.html', success=False, message=message)
                
        best_match_name = None
        best_similarity = float('inf') 

        for i, stored_embedding in enumerate(known_face_embeddings_list):
            try:
                if face_embedding.shape != stored_embedding.shape: continue 
                similarity = cosine(face_embedding, stored_embedding)
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_match_name = known_face_names[i]
            except Exception as e:
                logging.error(f"Error comparing cached embedding {i} in attendance: {e}")
                continue
        
        if best_match_name and best_similarity < RECOGNITION_THRESHOLD:
            student_name = best_match_name
            student_branch = known_face_data.get(student_name, {}).get('branch', 'N/A')
            logging.info(f"Attendance match found: {student_name} with cosine distance {best_similarity:.4f}")

            today_str = datetime.now(india_tz).strftime('%Y-%m-%d')
            existing_attendance_student = mongo.db.attendance.find_one({
                'name': student_name,
                'time': {'$regex': f'^{today_str}'}
            })

            if existing_attendance_student:
                logging.info(f"Attendance attempt by {student_name}, but already marked today.")
                
                return render_template('already_marked.html', student_name=student_name)
            else:
                
                current_time_str = datetime.now(india_tz).strftime('%Y-%m-%d %H:%M:%S')
                try:
                    mongo.db.attendance.insert_one({
                        'name': student_name,
                        'branch': student_branch,
                        'time': current_time_str
                    })
                    logging.info(f"Attendance successfully recorded for {student_name} at {current_time_str}")
                    return render_template('attendance_result.html', success=True, student_name=student_name, time=current_time_str)
                except Exception as e:
                     logging.error(f"Database error during attendance marking for {student_name}: {e}")
                     return render_template('attendance_result.html', success=False, message="Database error occurred while marking attendance.")
        else:
            
            log_msg = f"No matching student found for attendance attempt."
            user_msg = "Could not find a matching face for attendance. Please try again."
            if best_match_name: 
                 log_msg += f" Closest: {best_match_name} (Dist: {best_similarity:.4f})"
                 user_msg = f"Face doesn't match registered students closely enough (Closest: {best_match_name}, Dist: {best_similarity:.4f}). Ensure good lighting and clear view."

            logging.warning(log_msg)
            return render_template('attendance_result.html', success=False, message=user_msg)

    
    if user_type == 'student':
         return render_template('attendance.html')
    else:
         
         return redirect(url_for('index' if user_type == 'admin' else 'login'))

@app.route('/students', methods=['GET'])
def students_view():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))
    try:
        
        students_cursor = mongo.db.students.find({}, {"name": 1, "branch": 1, "registered_at": 1})
        students_list = list(students_cursor)
        
        for student in students_list:
             if 'registered_at' in student and isinstance(student['registered_at'], datetime):
                 
                 student['registered_at_str'] = student['registered_at'].strftime('%Y-%m-%d %H:%M:%S')
        return render_template('students.html', students=students_list)
    except Exception as e:
        logging.error(f"Error fetching students list: {e}")
        return "Error loading student data.", 500

@app.route('/present')
def present_view():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))
    try:
        
        attendance_records = list(mongo.db.attendance.find().sort('time', -1))
        logging.debug(f"Fetched {len(attendance_records)} attendance records.")
        return render_template('present.html', attendance=attendance_records)
    except Exception as e:
        logging.error(f"Error fetching attendance records: {e}")
        return "Error loading attendance data.", 500

@app.route('/reset_attendance', methods=['POST'])
def reset_attendance_route():
    if session.get('user_type') != 'admin':
        return jsonify({"message": "Unauthorized"}), 403
    try:
        result = mongo.db.attendance.delete_many({})
        deleted_count = result.deleted_count
        logging.info(f"Reset attendance requested. {deleted_count} records deleted.")
        return jsonify({"message": f"Attendance has been reset. {deleted_count} records deleted."}), 200
    except Exception as e:
         logging.error(f"Error resetting attendance: {e}")
         return jsonify({"message": "An error occurred while resetting attendance."}), 500

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
            
            filename = secure_filename("admin_upload_" + file.filename)
            try:
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                logging.info(f"Admin uploaded photo: {filename} to {save_path}")
                
                return redirect(url_for('index')) 
            except Exception as e:
                 logging.error(f"Error saving admin uploaded photo '{filename}': {e}")
                 return "Error saving uploaded file.", 500

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


# --- New Route for Seen Log ---
@app.route('/seen_log')
def seen_log_view():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))
    try:
        # Fetch logs, sort by timestamp descending, limit for performance if needed
        log_records = list(mongo.db.seen_log.find().sort('timestamp', -1).limit(200)) # Limit to last 200 entries
        logging.debug(f"Fetched {len(log_records)} seen log records.")
        # Format timestamp for display
        india_tz = pytz.timezone('Asia/Kolkata')
        for record in log_records:
            if isinstance(record.get('timestamp'), datetime):
                 record['timestamp_str'] = record['timestamp'].astimezone(india_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                 record['timestamp_str'] = "Invalid Date" # Handle potential non-datetime values

        return render_template('seen_log.html', logs=log_records)
    except Exception as e:
        logging.error(f"Error fetching seen log records: {e}")
        return "Error loading seen log data.", 500
# --- End New Route ---


# --- New Route to Reset Seen Log ---
@app.route('/reset_seen_log', methods=['POST'])
def reset_seen_log():
    if session.get('user_type') != 'admin':
        logging.warning("Unauthorized attempt to reset seen log.")
        flash("Unauthorized access.", "error")
        return redirect(url_for('login')) # Or perhaps seen_log_view with error

    try:
        seen_log_collection = mongo.db.seen_log
        result = seen_log_collection.delete_many({})
        deleted_count = result.deleted_count
        logging.info(f"Seen log reset requested by admin. {deleted_count} records deleted.")
        flash(f"Seen log successfully reset. {deleted_count} records removed.", "success")
    except Exception as e:
        logging.error(f"Error resetting seen log: {e}")
        flash("An error occurred while resetting the seen log.", "error")

    return redirect(url_for('seen_log_view'))
# --- End Reset Seen Log Route ---


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
