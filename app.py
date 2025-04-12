from flask import Flask, render_template, request, redirect, url_for, jsonify, session, Response, flash
import os
import logging

# Ensure debug messages are shown in the console
logging.basicConfig(level=logging.DEBUG)
from datetime import datetime, timedelta, time
import pytz
from collections import defaultdict
import cv2
import numpy as np
import io
from io import BytesIO
import logging
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import base64
import urllib.parse
from scipy.spatial.distance import cosine
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
import json
import openpyxl
from markupsafe import Markup

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
CONFIG_FILE = 'config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {CONFIG_FILE}. Using defaults.")
            return {}
        except Exception as e:
            logging.error(f"Error reading config file {CONFIG_FILE}: {e}")
            return {}
    return {}

def save_config(config_data):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        logging.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        logging.error(f"Error writing config file {CONFIG_FILE}: {e}")
        # flash("Error saving configuration.", "error")

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

CAMERAS_FILE = 'cameras.json'

def load_cameras():
    cameras = []
    if os.path.exists(CAMERAS_FILE):
        try:
            with open(CAMERAS_FILE, 'r') as f:
                cameras = json.load(f)
        except Exception as e:
            logging.error(f"Error reading cameras file {CAMERAS_FILE}: {e}")
            cameras = []
    # Ensure PC Webcam is always present
    has_webcam = any(
        (str(cam.get('id')) == "0" or str(cam.get('rtsp_url')) == "0" or cam.get('rtsp_url') == 0)
        for cam in cameras
    )
    if not has_webcam:
        cameras.insert(0, {
            "id": 0,
            "name": "PC Webcam",
            "ip": "",
            "port": "",
            "username": "",
            "password": "",
            "rtsp_url": 0
        })
    return cameras

def save_cameras(cameras):
    try:
        with open(CAMERAS_FILE, 'w') as f:
            json.dump(cameras, f, indent=4)
        logging.info(f"Cameras saved to {CAMERAS_FILE}")
    except Exception as e:
        logging.error(f"Error writing cameras file {CAMERAS_FILE}: {e}")

@app.route('/configure', methods=['GET', 'POST'])
def configure():
    # Admin check
    if session.get('user_type') != 'admin':
        # flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    config = load_config()
    cameras = load_cameras()
    if request.method == 'POST':
        # Handle Academic Year
        academic_year = request.form.get('academic_year', '')
        if academic_year:
            config['academic_year'] = academic_year
        else:
            config['academic_year'] = ''

        # Robust validation for unknown_log_interval_seconds
        unknown_interval_str = request.form.get('unknown_log_interval_seconds')
        unknown_interval_flash = {'message': "", 'category': "info"}

        if unknown_interval_str is not None:
            try:
                unknown_interval_val = int(unknown_interval_str)
                if unknown_interval_val >= 1:
                    config['unknown_log_interval_seconds'] = unknown_interval_val
                    unknown_interval_flash['message'] = f"Unknown face log interval successfully set to {unknown_interval_val} seconds."
                    unknown_interval_flash['category'] = "success"
                    logging.info(f"Configuration updated: unknown_log_interval_seconds = {unknown_interval_val}")
                else:
                    unknown_interval_flash['message'] = "Unknown face log interval must be 1 second or greater. Value not saved."
                    unknown_interval_flash['category'] = "warning"
                    logging.warning(f"Invalid unknown_log_interval_seconds value received: {unknown_interval_val}. Not saved.")
            except (ValueError, TypeError):
                unknown_interval_flash['message'] = "Invalid input for unknown face log interval. Please enter a whole number (e.g., 60). Value not saved."
        else:
            # Fallback to previous logic if not provided
            config['unknown_log_interval_seconds'] = int(request.form.get('unknown_log_interval_seconds', config.get('unknown_log_interval_seconds', 60)))

        # Handle other settings update
        config['stop_time_enabled'] = bool(request.form.get('enable_stop_time'))
        config['stop_time'] = request.form.get('stop_time', config.get('stop_time', '14:56'))
        config['unknown_face_timeout'] = int(request.form.get('unknown_face_timeout', config.get('unknown_face_timeout', 3)))
        save_config(config)

        # Handle camera add
        if 'add_camera' in request.form:
            name = request.form.get('camera_name')
            ip = request.form.get('camera_ip')
            port = request.form.get('camera_port')
            username = request.form.get('camera_username')
            password = request.form.get('camera_password')
            rtsp_url = request.form.get('camera_rtsp_url')
            if name and ip and port and username and password and rtsp_url:
                new_id = max([c.get('id', 0) for c in cameras], default=0) + 1
                cameras.append({
                    "id": new_id,
                    "name": name,
                    "ip": ip,
                    "port": int(port),
                    "username": username,
                    "password": password,
                    "rtsp_url": rtsp_url
                })
                save_cameras(cameras)
                flash(Markup(f"Camera <b>{name}</b> added."), "success")
            else:
                flash("All camera fields are required.", "danger")

        # Handle camera remove
        if 'remove_camera' in request.form:
            remove_id = int(request.form.get('remove_camera'))
            cameras = [c for c in cameras if c['id'] != remove_id]
            save_cameras(cameras)
            flash("Camera removed.", "success")

        return redirect(url_for('configure'))

    # GET: Render page
    return render_template(
        'configure.html',
        stop_time_enabled=config.get('stop_time_enabled', False),
        current_stop_time=config.get('stop_time', '14:56'),
        unknown_face_timeout=config.get('unknown_face_timeout', 3),
        unknown_log_interval_seconds=config.get('unknown_log_interval_seconds', 60),
        cameras=cameras,
        academic_year=config.get('academic_year', '')
    )

RECOGNITION_THRESHOLD = 0.36

def load_known_faces():
    global known_face_data, known_face_embeddings_list, known_face_names
    logging.info("Loading known faces from database...")
    known_face_data = {}
    temp_embeddings_list = []
    temp_names_list = []
    try:
        users_cursor = mongo.db.students.find({}, {"name": 1, "branch": 1, "role": 1, "face_embedding": 1})
        count = 0
        for user_doc in users_cursor:
            if 'face_embedding' in user_doc and user_doc['face_embedding'] and 'name' in user_doc:
                try:
                    embedding_np = np.array(user_doc['face_embedding']).astype(np.float32)
                    if embedding_np.shape == (128,):
                        known_face_data[user_doc['name']] = {
                            'embedding': embedding_np,
                            'branch': user_doc.get('branch'),
                            'role': user_doc.get('role', 'Unknown')
                        }
                        temp_embeddings_list.append(embedding_np)
                        temp_names_list.append(user_doc['name'])
                        count += 1
                    else:
                        logging.warning(f"Invalid embedding shape {embedding_np.shape} for user {user_doc.get('name', 'N/A')}. Expected (128,). Skipping.")

                except Exception as e:
                    logging.warning(f"Could not process embedding for user {user_doc.get('name', 'N/A')}: {e}")
            else:
                logging.warning(f"User data incomplete or missing face_embedding: {user_doc.get('_id')}")

        known_face_embeddings_list = temp_embeddings_list
        known_face_names = temp_names_list
        logging.info(f"Loaded {count} known faces (embeddings) into memory cache.")

    except Exception as e:
        logging.error(f"Error loading known faces from database: {e}")

def detect_and_encode_face(image_np):
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
        flash(f"Face cache refreshed. Loaded {len(known_face_data)} faces.", "success")
        return jsonify({"message": f"Face cache refreshed. Loaded {len(known_face_data)} faces."}), 200
    except Exception as e:
        logging.error(f"Error during manual face cache refresh: {e}")
        # flash("Error refreshing face cache.", "error")
        return jsonify({"message": "Error refreshing face cache."}), 500

@app.route('/')
def index():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/live_feed')
def live_feed():
    if session.get('user_type') != 'admin':
        # flash("Unauthorized access to live feed.", "warning")
        return redirect(url_for('login'))
    cameras = load_cameras()
    return render_template('live_feed.html', cameras=cameras)


def generate_frames(source):
    config = load_config()
    stop_time_enabled = config.get('stop_time_enabled', False)
    stop_time_str = config.get('stop_time')
    stop_time_obj = None
    india_tz = pytz.timezone('Asia/Kolkata')

    if stop_time_enabled and stop_time_str:
        try:
            stop_time_obj = datetime.strptime(stop_time_str, '%H:%M').time()
            logging.info(f"Live feed monitoring enabled to stop at {stop_time_str}")
        except ValueError:
            logging.warning(f"Invalid stop_time format '{stop_time_str}' in config. Ignoring stop time despite being enabled.")
            stop_time_obj = None
    elif stop_time_enabled:
        logging.warning("Stop time is enabled in config, but no valid time is set.")
    else:
        logging.info("Automatic live feed stop time is disabled.")


    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Cannot open camera 0")
        return

    frame_count = 0
    process_every_n_frames = 3
    last_known_faces = {}
    seen_log_collection = mongo.db.seen_log
    last_log_times = {}
    log_interval = timedelta(minutes=1)
    last_unknown_log_time = None
    unknown_log_interval_seconds = config.get('unknown_log_interval_seconds', 60)
    unknown_log_interval = timedelta(seconds=unknown_log_interval_seconds)
    logging.info(f"Unknown face log interval set to: {unknown_log_interval_seconds} seconds.")


    while True:
        success, frame = cap.read()
        if not success:
            logging.warning("Failed to read frame from camera.")
            break

        if stop_time_enabled and stop_time_obj:
            current_time_india = datetime.now(india_tz).time()
            if current_time_india >= stop_time_obj:
                logging.info(f"Current time {current_time_india} reached or passed enabled stop time {stop_time_obj}. Stopping live feed.")
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
                current_time_utc = datetime.utcnow()
                recognized_user_name = None

                if face_embedding is not None:
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

                    if best_match_name and best_similarity < RECOGNITION_THRESHOLD:
                        recognized_user_name = best_match_name
                        logging.info(f"Live Feed Match: {recognized_user_name} (Cosine Dist: {best_similarity:.4f})")
                        status_text = f"Recognized: {recognized_user_name}"
                        status_color = (0, 255, 0)

                    elif best_match_name:
                        status_text = f"Low Match: {best_match_name} ({best_similarity:.2f})"
                        status_color = (0, 165, 255)
                    else:
                        status_text = "Unknown Face"
                        status_color = (255, 0, 0)

                    current_detected_faces[0] = {'box': box, 'status': status_text, 'color': status_color}

                else:
                    status_text = "Processing Error"
                    status_color = (255, 0, 255)
                    current_detected_faces[0] = {'box': box, 'status': status_text, 'color': status_color}

                if box:
                    log_entry = {
                        'timestamp': current_time_utc,
                        'status_at_log': status_text
                    }

                    if recognized_user_name:
                        now_utc = current_time_utc
                        last_logged = last_log_times.get(recognized_user_name)
                        if last_logged is None or (now_utc - last_logged) >= log_interval:
                            log_entry['type'] = 'known_sighting'
                            log_entry['name'] = recognized_user_name
                            try:
                                seen_log_collection.insert_one(log_entry)
                                last_log_times[recognized_user_name] = now_utc
                                logging.info(f"Logged sighting of known face: {recognized_user_name} (Interval passed or first time)")
                            except Exception as db_err:
                                logging.error(f"Error inserting known sighting log for {recognized_user_name}: {db_err}")

                    elif status_text.startswith("Low Match") or status_text == "Processing Error":
                        now_utc = current_time_utc
                        if last_unknown_log_time is None or (now_utc - last_unknown_log_time) >= unknown_log_interval:
                            if status_text == "Processing Error":
                                log_entry['type'] = 'processing_error_sighting'
                                log_entry['error_details'] = "Failed to generate face embedding"
                            else:
                                log_entry['type'] = 'unknown_sighting'

                            try:
                                (x, y, w, h) = box
                                if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= processing_frame.shape[1] and y + h <= processing_frame.shape[0]:
                                    cropped_face = processing_frame[y:y+h, x:x+w]
                                    if cropped_face.size > 0:
                                        _, buffer = cv2.imencode('.jpg', cropped_face)
                                        log_entry['face_image_base64'] = base64.b64encode(buffer).decode('utf-8')
                                    else: logging.warning("Cropped unknown/low match face is empty, logging without image.")
                                else: logging.warning(f"Invalid box coordinates for cropping unknown/low match face: {box}, frame shape: {processing_frame.shape}")
                            except Exception as img_err:
                                 logging.error(f"Error processing image for unknown/low match sighting log: {img_err}")
                                 log_entry.pop('face_image_base64', None)

                            try:
                                seen_log_collection.insert_one(log_entry)
                                last_unknown_log_time = now_utc
                                logging.info(f"Logged unknown/low match face sighting ({status_text}) {'with image' if 'face_image_base64' in log_entry else 'without image'}. Interval passed.")
                            except Exception as db_err:
                                logging.error(f"Error inserting unknown/low match sighting log into DB: {db_err}")

            last_known_faces = current_detected_faces if box else {}
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
    camera_id = request.args.get('camera_id', type=int)
    cameras = load_cameras()
    camera = None
    if camera_id is not None:
        for cam in cameras:
            if cam.get('id') == camera_id:
                camera = cam
                break
        if not camera:
            return Response("Camera not found", status=404)
        rtsp_url = camera.get('rtsp_url')
        # Convert "0", "1", etc. to int for local webcams
        if isinstance(rtsp_url, str) and rtsp_url.isdigit():
            rtsp_url = int(rtsp_url)
        return Response(generate_frames(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    # Default: fallback to first camera or local camera 0
    if cameras:
        rtsp_url = cameras[0].get('rtsp_url')
        # Convert "0", "1", etc. to int for local webcams
        if isinstance(rtsp_url, str) and rtsp_url.isdigit():
            rtsp_url = int(rtsp_url)
        return Response(generate_frames(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    # If no cameras configured, fallback to local camera 0
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    print("[REGISTER DEBUG] register() route entered")
    print("[REGISTER DEBUG] register() called. Method:", request.method, "Form:", dict(request.form), "Files:", dict(request.files))
    if session.get('user_type') != 'admin':
        # flash("Only admins can register new users.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form.get('name')
        role = request.form.get('role')
        role = role.lower() if role else None
        branch = request.form.get('branch')
        academic_year = request.form.get('academic_year')
        photo_data = request.form.get('photo')
        uploaded_file = request.files.get('uploadPhoto')

        allowed_roles = ['student', 'staff', 'others']
        if not role or role not in allowed_roles:
            logging.debug(f"[REGISTER DEBUG] Invalid or missing role: {role}, allowed: {allowed_roles}")
            print(f"[REGISTER DEBUG] Invalid or missing role: {role}, allowed: {allowed_roles}")
            flash(f"Invalid or missing role. Must be one of: {', '.join(allowed_roles)}.", "error")
            print("[REGISTER DEBUG] Returning 400: Invalid or missing role")
            print("[REGISTER DEBUG] Returning 400: Name is required")
            return render_template('register.html'), 400

        if not name:
            logging.debug(f"[REGISTER DEBUG] Name is missing. Form data: {request.form}")
            print(f"[REGISTER DEBUG] Name is missing. Form data: {request.form}")
            # flash("Name is required.", "error")
            print("[REGISTER DEBUG] Returning 400: Branch is required for students")
            print("[REGISTER DEBUG] Returning 400: Academic year invalid or missing")
            return render_template('register.html'), 400

        # Updated validation: Branch required for students/staff, Academic Year required for students
        if role == 'student':
            if not branch:
                logging.debug(f"[REGISTER DEBUG] Branch is missing for student. Form data: {request.form}")
                print(f"[REGISTER DEBUG] Branch is missing for student. Form data: {request.form}")
                # flash("Branch is required for students.", "error")
                print("[REGISTER DEBUG] Returning 400: Branch/Department is required for staff")
                return render_template('register.html'), 400
            valid_academic_years = ['1st', '2nd', '3rd']
            if not academic_year or academic_year not in valid_academic_years:
                print(f"[REGISTER DEBUG] Academic year invalid or missing: {academic_year}. Form data: {request.form}")
                # flash("Valid Academic Year (1st, 2nd, or 3rd) is required for students.", "error")
                return render_template('register.html'), 400
        elif role == 'staff':
            if not branch:
                logging.debug(f"[REGISTER DEBUG] Branch is missing for staff. Form data: {request.form}")
                print(f"[REGISTER DEBUG] Branch is missing for staff. Form data: {request.form}")
                # flash("Branch/Department is required for staff.", "error")
                return render_template('register.html'), 400
            academic_year = None

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
                # flash("Invalid photo data from webcam.", "error")
                print("[REGISTER DEBUG] Returning 400: No photo provided")
                return render_template('register.html'), 400
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
                flash(f"Could not process uploaded file: {e}", "error")
                if file_path and os.path.exists(file_path):
                     try: os.remove(file_path)
                     except OSError as rm_err: logging.error(f"Error removing temporary file {file_path}: {rm_err}")
                return render_template('register.html'), 400
        else:
            logging.debug(f"[REGISTER DEBUG] No photo provided. Form data: {request.form}, Files: {request.files}")
            print(f"[REGISTER DEBUG] No photo provided. Form data: {request.form}, Files: {request.files}")
            # flash("No photo provided (either webcam capture or file upload is required).", "error")
            print("[REGISTER DEBUG] Returning 400: User already exists")
            return render_template('register.html'), 400

        if image is None:
            logging.error("Image data is None after loading attempts.")
            if file_path and os.path.exists(file_path):
                try: os.remove(file_path)
                except OSError as rm_err: logging.error(f"Error removing file {file_path} after load fail: {rm_err}")
            # flash("Failed to load image data.", "error")
            logging.debug(f"[REGISTER DEBUG] Image data is None after loading attempts. File path: {file_path}")
            print(f"[REGISTER DEBUG] Image data is None after loading attempts. File path: {file_path}")
            print("[REGISTER DEBUG] Returning 500: Image data is None after loading attempts")
            return render_template('register.html'), 500

        existing_user = mongo.db.students.find_one({'name': name})
        if existing_user:
            logging.warning(f"Registration attempt for existing user: {name}")
            flash(f"User '{name}' is already registered!", "warning")
            if file_path and os.path.exists(file_path):
                try: os.remove(file_path)
                except OSError as e: logging.error(f"Error removing upload file {file_path} for existing user: {e}")
            logging.debug(f"[REGISTER DEBUG] User already exists: {name}")
            print(f"[REGISTER DEBUG] User already exists: {name}")
            print("[REGISTER DEBUG] Returning 400: Face detection/embedding failed")
            return render_template('register.html'), 400

        logging.info(f"Processing registration for {name} from {image_source}.")
        face_embedding, box = detect_and_encode_face(image)

        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except OSError as e: logging.error(f"Error removing upload file {file_path} after processing: {e}")

        if face_embedding is None:
            logging.warning(f"No face detected or embedding failed for registration image of {name}.")
            if box:
                message = "Face detected, but could not process it for registration. Try a clearer image or different angle."
            else:
                message = "No face detected in the provided image. Please ensure the face is clear and well-lit."
            flash(message, "error")
            logging.debug(f"[REGISTER DEBUG] Face detection/embedding failed for {name}. Face box: {box}")
            print(f"[REGISTER DEBUG] Face detection/embedding failed for {name}. Face box: {box}")
            return render_template('register.html'), 400

        try:
            embedding_list = face_embedding.tolist()
            user_data = {
                'name': name,
                'role': role,
                'face_embedding': embedding_list,
                'registered_at': datetime.utcnow() # Store as UTC
            }
            # Conditionally add branch and academic_year
            if role == 'student':
                user_data['branch'] = branch
                user_data['academic_year'] = academic_year
            elif role == 'staff':
                user_data['branch'] = branch
                user_data['academic_year'] = None
            else:
                user_data['branch'] = None
                user_data['academic_year'] = None

            mongo.db.students.insert_one(user_data)
            logging.info(f"Successfully registered user: {name} with role: {role}")
            flash(f"User '{name}' ({role}) registered successfully!", "success")
            load_known_faces()
            return redirect(url_for('view_students'))

        except Exception as e:
            logging.error(f"Database error during registration for {name}: {e}")
            logging.debug(f"[REGISTER DEBUG] Database error during registration for {name}: {e}")
            print(f"[REGISTER DEBUG] Database error during registration for {name}: {e}")
            # flash("Database error occurred during registration.", "error")
            print("[REGISTER DEBUG] Returning 500: Database error during registration")
            return render_template('register.html'), 500

    return render_template('register.html')


@app.route('/view_students', methods=['GET'])
def view_students():
    if session.get('user_type') != 'admin':
        # flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        search_query = request.args.get('search', '')
        role_filter = request.args.get('role_filter', '')
        branch_filter = request.args.get('branch_filter', '')
        academic_year_filter = request.args.get('academic_year_filter', '')
        sort_by = request.args.get('sort_by', 'name')
        sort_order = request.args.get('sort_order', 'asc')
        india_tz = pytz.timezone('Asia/Kolkata')

        try:
            distinct_branches = mongo.db.students.distinct('branch', {'branch': {'$nin': [None, '']}, 'role': {'$in': ['student', 'staff']}})
            distinct_branches.sort()
        except Exception as db_err:
            logging.error(f"Error fetching distinct branches: {db_err}")
            # flash("Error loading branch filter options.", "error")
            distinct_branches = []

        query_conditions = []
        if search_query:
            regex = {'$regex': search_query, '$options': 'i'}
            query_conditions.append({'$or': [{'name': regex}, {'branch': regex}]})

        if role_filter:
            query_conditions.append({'role': role_filter})

        if branch_filter:
            query_conditions.append({'branch': branch_filter})
        if academic_year_filter:
            query_conditions.append({'academic_year': academic_year_filter})

        query = {}
        if query_conditions:
            query['$and'] = query_conditions

        mongo_sort_order = 1 if sort_order == 'asc' else -1
        valid_sort_fields = ['name', 'role', 'registered_at', 'branch']
        if sort_by not in valid_sort_fields:
            sort_by = 'name'

        students_cursor = mongo.db.students.find(query, {"name": 1, "branch": 1, "role": 1, "registered_at": 1, "_id": 1, "academic_year": 1}) \
                                         .sort(sort_by, mongo_sort_order)
        students_list = list(students_cursor)

        for student in students_list:
            reg_at_utc = student.get('registered_at')
            if isinstance(reg_at_utc, datetime):
                 # Assume stored as UTC, make aware, convert to Kolkata
                 reg_at_kolkata = reg_at_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                 student['registered_at_str'] = reg_at_kolkata.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                 student['registered_at_str'] = "N/A"
            student['_id_str'] = str(student['_id'])

        logging.info(f"Distinct branches being passed to template: {distinct_branches}")

        return render_template('view_students.html',
                               students=students_list,
                               search_query=search_query,
                               role_filter=role_filter,
                               branch_filter=branch_filter,
                               academic_year_filter=academic_year_filter,
                               distinct_branches=distinct_branches,
                               sort_by=sort_by,
                               sort_order=sort_order)

    except Exception as e:
        logging.exception(f"Error fetching students list for view:")
        try:
            distinct_branches = mongo.db.students.distinct('branch', {'branch': {'$nin': [None, '']}, 'role': {'$in': ['student', 'staff']}})
            distinct_branches.sort()
        except Exception as db_err:
             logging.error(f"Error fetching distinct branches during exception handling: {db_err}")
             distinct_branches = []

        # flash("Error loading student data. Please check logs or contact support.", "error")
        return render_template('view_students.html',
                               students=[],
                               search_query=request.args.get('search', ''),
                               role_filter=request.args.get('role_filter', ''),
                               branch_filter=request.args.get('branch_filter', ''),
                               academic_year_filter=request.args.get('academic_year_filter', ''),
                               distinct_branches=distinct_branches,
                               sort_by=request.args.get('sort_by', 'name'),
                               sort_order=request.args.get('sort_order', 'asc')), 200


@app.route('/delete_student/<student_id>', methods=['POST'])
def delete_student(student_id):
    if session.get('user_type') != 'admin':
         # flash("Unauthorized action.", "error")
         return redirect(url_for('view_students'))

    try:
        oid = ObjectId(student_id)
    except Exception:
        # flash("Invalid student ID format.", "error")
        return redirect(url_for('view_students'))

    try:
        student_to_delete = mongo.db.students.find_one({'_id': oid}, {'name': 1})
        if not student_to_delete:
            # flash("Student not found for deletion.", "error")
            return redirect(url_for('view_students')), 404

        student_name = student_to_delete.get('name', 'Unknown')
        result = mongo.db.students.delete_one({'_id': oid})

        if result.deleted_count > 0:
            logging.info(f"Successfully deleted student: {student_name} (ID: {student_id})")
            flash(f"Student '{student_name}' deleted successfully.", "success")
            load_known_faces()
            try:
                log_result = mongo.db.seen_log.delete_many({'name': student_name, 'type': 'known_sighting'})
                logging.info(f"Removed {log_result.deleted_count} sighting logs for deleted student {student_name}.")
            except Exception as log_e:
                logging.error(f"Error removing sighting logs for {student_name}: {log_e}")
        else:
            logging.warning(f"Student with ID {student_id} found but deletion failed.")
            # flash("Student found but could not be deleted.", "error")

    except Exception as e:
        logging.error(f"Error deleting student {student_id}: {e}")
        # flash("An error occurred while trying to delete the student.", "error")

    return redirect(url_for('view_students'))


# Removed duplicate configure route and function (merged into single definition above)


    current_stop_time = config.get('stop_time', '')
    stop_time_enabled = config.get('stop_time_enabled', False)

    unknown_face_timeout = config.get('unknown_face_timeout', 5)
    academic_year = config.get('academic_year', '')
    return render_template('configure.html',
                           current_stop_time=current_stop_time,
                           stop_time_enabled=stop_time_enabled,
                           unknown_log_interval_seconds=unknown_log_interval_seconds,
                           unknown_face_timeout=unknown_face_timeout,
                           academic_year=academic_year)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_type') == 'admin':
        return redirect(url_for('index'))

    if request.method == 'POST':
        password = request.form.get('admin_password')
        admin_pass = os.environ.get("ADMIN_PASSWORD", "0000")
        if password == admin_pass:
            session['user_type'] = 'admin'
            session['user_name'] = 'Admin'
            logging.info("Admin login successful.")
            # flash("Admin login successful.", "success")
            return redirect(url_for('index'))
        else:
            logging.warning("Invalid admin password attempt.")
            # flash("Invalid admin credentials.", "error")
            return render_template('login.html'), 401

    return render_template('login.html')


@app.route('/logout')
def logout():
    user_name = session.get('user_name', 'User')
    session.pop('user_type', None)
    session.pop('user_name', None)
    logging.info(f"User '{user_name}' logged out.")
    # flash("You have been logged out.", "info")
    return redirect(url_for('login'))


@app.before_request
def restrict_access():
    if request.endpoint in ['login', 'static', 'logout']:
        return None

    if 'user_type' not in session:
        # flash("Please log in to access this page.", "info")
        return redirect(url_for('login'))

    if session.get('user_type') == 'admin':
        return None

    # flash("Admin access required. Please log in.", "warning")
    return redirect(url_for('login'))


@app.route('/seen_log')
def seen_log_view():
    if session.get('user_type') != 'admin':
        # flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))
    try:
        log_records_cursor = mongo.db.seen_log.find().sort('timestamp', -1)
        log_records = list(log_records_cursor)
        logging.debug(f"Fetched {len(log_records)} seen log records.")

        grouped_logs = defaultdict(list)
        unknown_logs = []
        india_tz = pytz.timezone('Asia/Kolkata')

        for record in log_records:
            ts_utc = record.get('timestamp')
            if isinstance(ts_utc, datetime):
                # Assume stored as UTC, make aware, convert to Kolkata
                ts_kolkata = ts_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                record['timestamp_str'] = ts_kolkata.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                record['timestamp_str'] = "Invalid Date"

            if record.get('type') == 'known_sighting':
                name = record.get('name')
                if name:
                    grouped_logs[name].append(record)
                else:
                    record['name'] = 'Unknown (Error)'
                    unknown_logs.append(record)
            elif record.get('type') == 'unknown_sighting' or record.get('type') == 'processing_error_sighting':
                 unknown_logs.append(record)

        grouped_logs_dict = dict(grouped_logs)

        return render_template('seen_log.html', grouped_logs=grouped_logs_dict, unknown_logs=unknown_logs)
    except Exception as e:
        logging.error(f"Error fetching or grouping seen log records: {e}")
        # flash("Error loading seen log data.", "error")
        return render_template('seen_log.html', grouped_logs={}, unknown_logs=[]), 500

@app.route('/unknown_captures')
def unknown_captures():
    if session.get('user_type') != 'admin':
        # flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))
    try:
        # Parse optional start_time and end_time from query params
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')
        time_filter = {}

        if start_time_str:
            try:
                start_dt = datetime.fromisoformat(start_time_str)
                time_filter['$gte'] = start_dt
            except ValueError:
                pass
        if end_time_str:
            try:
                end_dt = datetime.fromisoformat(end_time_str)
                time_filter['$lte'] = end_dt
            except ValueError:
                pass
                print(f"[DEBUG] Invalid end time format received: {end_time_str}. Expected ISO format.")

        query = {"type": {"$in": ["unknown_sighting", "processing_error_sighting"]}}
        if time_filter:
            query['timestamp'] = time_filter

        log_records_cursor = mongo.db.seen_log.find(query).sort('timestamp', -1)
        log_records = list(log_records_cursor)
        logging.debug(f"Fetched {len(log_records)} unknown capture records with filter {time_filter}")

        grouped_unknowns = defaultdict(list)
        india_tz = pytz.timezone('Asia/Kolkata')

        for record in log_records:
            ts_utc = record.get('timestamp')
            if isinstance(ts_utc, datetime):
                ts_kolkata = ts_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                record['timestamp_str'] = ts_kolkata.strftime('%Y-%m-%d %H:%M:%S %Z')
                date_str = ts_kolkata.strftime('%Y-%m-%d')
            else:
                record['timestamp_str'] = "Invalid Date"
                date_str = "Unknown Date"

            grouped_unknowns[date_str].append(record)

        grouped_unknowns_dict = dict(grouped_unknowns)

        return render_template('unknown_captures.html', grouped_unknowns=grouped_unknowns_dict)
    except Exception as e:
        logging.error(f"Error fetching or grouping unknown captures: {e}")
        # flash("Error loading unknown captures data.", "error")
        return render_template('unknown_captures.html', grouped_unknowns={}), 500

@app.route('/reset_seen_log', methods=['POST'])
def reset_seen_log():
    if session.get('user_type') != 'admin':
        logging.warning("Unauthorized attempt to reset seen log.")
        # flash("Unauthorized action.", "error")
        return redirect(url_for('login'))

    try:
        seen_log_collection = mongo.db.seen_log
        result = seen_log_collection.delete_many({})
        deleted_count = result.deleted_count
        logging.info(f"Seen log reset requested by admin. {deleted_count} records deleted.")
    except Exception as e:
        logging.error(f"Error resetting seen log: {e}")
        # flash("An error occurred while resetting the seen log.", "error")

    return redirect(url_for('seen_log_view'))

@app.route('/get_attendance')
def get_attendance():
    if session.get('user_type') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now_utc = datetime.utcnow()
        now_india = now_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)

        # Get start/end of day in UTC based on Kolkata's current date
        start_of_day_kolkata = india_tz.localize(datetime.combine(now_india.date(), time.min))
        end_of_day_kolkata = india_tz.localize(datetime.combine(now_india.date(), time.max))
        start_of_day_utc = start_of_day_kolkata.astimezone(pytz.utc)
        end_of_day_utc = end_of_day_kolkata.astimezone(pytz.utc)

        logging.info(f"Fetching attendance logs between {start_of_day_utc} (UTC) and {end_of_day_utc} (UTC)")

        log_records_cursor = mongo.db.seen_log.find({
            'type': 'known_sighting',
            'timestamp': {
                '$gte': start_of_day_utc,
                '$lte': end_of_day_utc
            }
        }).sort('timestamp', 1)

        attendance_data = defaultdict(lambda: {'timestamps_utc': []})

        for record in log_records_cursor:
            name = record.get('name')
            timestamp_utc = record.get('timestamp')
            if name and isinstance(timestamp_utc, datetime):
                 attendance_data[name]['timestamps_utc'].append(timestamp_utc)

        final_attendance = {}
        for name, data in attendance_data.items():
            if data['timestamps_utc']:
                timestamps_utc = sorted(data['timestamps_utc'])
                arriving_time_utc = timestamps_utc[0]
                leaving_time_utc = timestamps_utc[-1]

                # Convert UTC times to Kolkata for final output
                arriving_time_kolkata = arriving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                leaving_time_kolkata = leaving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)

                final_attendance[name] = {
                    'name': name,
                    'arriving_time': arriving_time_kolkata.strftime('%Y-%m-%d %H:%M:%S'),
                    'leaving_time': leaving_time_kolkata.strftime('%Y-%m-%d %H:%M:%S')
                }

        logging.info(f"Processed attendance for {len(final_attendance)} individuals.")
        return jsonify(list(final_attendance.values()))

    except Exception as e:
        logging.error(f"Error calculating attendance: {e}")
        return jsonify({"error": "Failed to calculate attendance"}), 500

@app.route('/attendance')
def attendance_page():
    if session.get('user_type') != 'admin':
        # flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now_utc = datetime.utcnow()
        now_india = now_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)

        # Get start/end of day in UTC based on Kolkata's current date
        start_of_day_kolkata = india_tz.localize(datetime.combine(now_india.date(), time.min))
        end_of_day_kolkata = india_tz.localize(datetime.combine(now_india.date(), time.max))
        start_of_day_utc = start_of_day_kolkata.astimezone(pytz.utc)
        end_of_day_utc = end_of_day_kolkata.astimezone(pytz.utc)

        logging.info(f"Fetching attendance logs for attendance page between {start_of_day_utc} (UTC) and {end_of_day_utc} (UTC)")

        log_records_cursor = mongo.db.seen_log.find({
            'type': 'known_sighting',
            'timestamp': {
                '$gte': start_of_day_utc,
                '$lte': end_of_day_utc
            }
        }).sort('timestamp', 1)

        attendance_data = defaultdict(lambda: {'timestamps_utc': []})
        user_details = {}

        # Use academic_year instead of role
        known_users = list(mongo.db.students.find({}, {"name": 1, "branch": 1, "academic_year": 1}))
        for user in known_users:
            user_details[user['name']] = {
                'branch': user.get('branch', 'N/A'),
                'academic_year': user.get('academic_year', 'N/A')
            }

        for record in log_records_cursor:
            name = record.get('name')
            timestamp_utc = record.get('timestamp')
            if name and isinstance(timestamp_utc, datetime):
                 attendance_data[name]['timestamps_utc'].append(timestamp_utc)

        # Default filters
        DEFAULT_BRANCH = "C.S.E."
        DEFAULT_ACADEMIC_YEAR = "1st"

        # Get filters from query params, fallback to defaults
        branch_filter = request.args.get('branch_filter', DEFAULT_BRANCH)
        academic_year_filter = request.args.get('academic_year_filter', DEFAULT_ACADEMIC_YEAR)

        final_attendance = []
        for name, data in attendance_data.items():
            if data['timestamps_utc']:
                timestamps_utc = sorted(data['timestamps_utc'])
                arriving_time_utc = timestamps_utc[0]
                leaving_time_utc = timestamps_utc[-1]

                # Convert UTC times to Kolkata for final output
                arriving_time_kolkata = arriving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                leaving_time_kolkata = leaving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)

                details = user_details.get(name, {'branch': 'N/A', 'academic_year': 'N/A'})
                # Apply filters
                if details['branch'] == branch_filter and details['academic_year'] == academic_year_filter:
                    final_attendance.append({
                        'name': name,
                        'branch': details['branch'],
                        'academic_year': details['academic_year'],
                        'arriving_time': arriving_time_kolkata.strftime('%H:%M:%S'),
                        'leaving_time': leaving_time_kolkata.strftime('%H:%M:%S'),
                        'date': arriving_time_kolkata.strftime('%Y-%m-%d')
                    })

        logging.info(f"Processed attendance for {len(final_attendance)} individuals for attendance page.")
        final_attendance.sort(key=lambda x: x['name'])

        return render_template('attendance.html', attendance_records=final_attendance, branch_filter=branch_filter, academic_year_filter=academic_year_filter)
    except Exception as e:
        logging.error(f"Error generating attendance page: {e}")
        # flash("Error loading attendance data.", "error")
        return render_template('attendance.html', attendance_records=[]), 500
        return render_template('attendance.html', attendance_records=[]), 500

@app.route('/export_attendance_excel')
def export_attendance_excel():
    if session.get('user_type') != 'admin':
        # flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now_utc = datetime.utcnow()
        now_india = now_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
        current_date_str = now_india.strftime('%Y-%m-%d')

        # Get start/end of day in UTC based on Kolkata's current date
        start_of_day_kolkata = india_tz.localize(datetime.combine(now_india.date(), time.min))
        end_of_day_kolkata = india_tz.localize(datetime.combine(now_india.date(), time.max))
        start_of_day_utc = start_of_day_kolkata.astimezone(pytz.utc)
        end_of_day_utc = end_of_day_kolkata.astimezone(pytz.utc)

        logging.info(f"Fetching attendance logs for Excel export between {start_of_day_utc} (UTC) and {end_of_day_utc} (UTC)")

        log_records_cursor = mongo.db.seen_log.find({
            'type': 'known_sighting',
            'timestamp': {
                '$gte': start_of_day_utc,
                '$lte': end_of_day_utc
            }
        }).sort('timestamp', 1)

        attendance_data = defaultdict(lambda: {'timestamps_utc': []})
        user_details = {}
        known_users = list(mongo.db.students.find({}, {"name": 1, "branch": 1, "role": 1}))
        for user in known_users:
            user_details[user['name']] = {'branch': user.get('branch', 'N/A'), 'role': user.get('role', 'N/A')}

        for record in log_records_cursor:
            name = record.get('name')
            timestamp_utc = record.get('timestamp')
            if name and isinstance(timestamp_utc, datetime):
                attendance_data[name]['timestamps_utc'].append(timestamp_utc)

        final_attendance = []
        for name, data in attendance_data.items():
            if data['timestamps_utc']:
                timestamps_utc = sorted(data['timestamps_utc'])
                arriving_time_utc = timestamps_utc[0]
                leaving_time_utc = timestamps_utc[-1]

                # Convert UTC times to Kolkata for final output
                arriving_time_kolkata = arriving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                leaving_time_kolkata = leaving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)

                details = user_details.get(name, {'branch': 'N/A', 'role': 'N/A'})
                final_attendance.append({
                    'name': name,
                    'branch': details['branch'],
                    'role': details['role'],
                    'arriving_time': arriving_time_kolkata.strftime('%H:%M:%S'),
                    'leaving_time': leaving_time_kolkata.strftime('%H:%M:%S'),
                    'date': arriving_time_kolkata.strftime('%Y-%m-%d')
                })
        final_attendance.sort(key=lambda x: x['name'])

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = f"Attendance {current_date_str}"

        headers = ["Name", "Branch", "Role", "Arriving Time", "Leaving Time", "Date"]
        sheet.append(headers)

        for record in final_attendance:
            sheet.append([
                record['name'],
                record['branch'],
                record['role'],
                record['arriving_time'],
                record['leaving_time'],
                record['date']
            ])

        excel_stream = BytesIO()
        workbook.save(excel_stream)
        excel_stream.seek(0)

        logging.info(f"Generated Excel export for {len(final_attendance)} records for date {current_date_str}.")

        return Response(
            excel_stream,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment;filename=attendance_{current_date_str}.xlsx'}
        )

    except Exception as e:
        logging.error(f"Error generating Excel export: {e}")
        # flash("Error generating Excel export.", "error")
        return redirect(url_for('attendance_page'))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)