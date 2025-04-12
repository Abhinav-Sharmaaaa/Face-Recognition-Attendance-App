from flask import Flask, render_template, request, redirect, url_for, jsonify, session, Response, flash
import os
import logging
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
import time as pytime

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
    default_config = {
        "stop_time": "18:00",
        "stop_time_enabled": False,
        "unknown_face_timeout": 5,
        "unknown_log_interval_seconds": 60,
        "academic_year": "",
        "process_every_n_frames": 10,
        "live_feed_sleep": 0.1,
        "jpeg_quality": 60
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
                return default_config
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {CONFIG_FILE}. Using defaults.")
            return default_config
        except Exception as e:
            logging.error(f"Error reading config file {CONFIG_FILE}: {e}. Using defaults.")
            return default_config
    logging.warning(f"Config file {CONFIG_FILE} not found. Using default values.")
    return default_config


def save_config(config_data):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        logging.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        logging.error(f"Error writing config file {CONFIG_FILE}: {e}")
        flash("Error saving configuration.", "error")

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
    if not cameras:
         cameras.append({
            "id": 0, "name": "PC Webcam", "ip": "", "port": "", "username": "", "password": "", "rtsp_url": 0, "disabled": False
         })
    else:
        has_webcam = any(
            (str(cam.get('id')) == "0" or str(cam.get('rtsp_url')) == "0" or cam.get('rtsp_url') == 0)
            for cam in cameras
        )
        if not has_webcam:
            cameras.insert(0, {
                "id": 0, "name": "PC Webcam", "ip": "", "port": "", "username": "", "password": "", "rtsp_url": 0, "disabled": False
            })

    for cam in cameras:
        if 'disabled' not in cam:
            cam['disabled'] = False

    return cameras

def save_cameras(cameras):
    try:
        with open(CAMERAS_FILE, 'w') as f:
            json.dump(cameras, f, indent=4)
        logging.info(f"Cameras saved to {CAMERAS_FILE}")
    except Exception as e:
        logging.error(f"Error writing cameras file {CAMERAS_FILE}: {e}")
        flash("Error saving camera configuration.", "error")

@app.route('/configure', methods=['GET', 'POST'])
def configure():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    config = load_config()
    cameras = load_cameras()

    if request.method == 'POST':
        try:
            config['stop_time_enabled'] = bool(request.form.get('enable_stop_time'))
            config['stop_time'] = request.form.get('stop_time', config.get('stop_time', '18:00'))

            try:
                timeout = int(request.form.get('unknown_face_timeout', config.get('unknown_face_timeout', 5)))
                config['unknown_face_timeout'] = max(1, timeout)
            except (ValueError, TypeError):
                flash("Invalid value for Unknown Face Timeout. Must be a number.", "warning")

            try:
                interval = int(request.form.get('unknown_log_interval_seconds', config.get('unknown_log_interval_seconds', 60)))
                config['unknown_log_interval_seconds'] = max(1, interval)
            except (ValueError, TypeError):
                flash("Invalid value for Unknown Log Interval. Must be a number.", "warning")

            try:
                n_frames = int(request.form.get('process_every_n_frames', config.get('process_every_n_frames', 10)))
                config['process_every_n_frames'] = max(1, n_frames)
            except (ValueError, TypeError):
                flash("Invalid value for Frame Skip. Must be a number.", "warning")

            try:
                feed_sleep = float(request.form.get('live_feed_sleep', config.get('live_feed_sleep', 0.1)))
                config['live_feed_sleep'] = max(0.0, feed_sleep)
            except (ValueError, TypeError):
                flash("Invalid value for Live Feed Sleep. Must be a number (e.g., 0.1).", "warning")

            try:
                quality = int(request.form.get('jpeg_quality', config.get('jpeg_quality', 60)))
                config['jpeg_quality'] = max(1, min(100, quality))
            except (ValueError, TypeError):
                flash("Invalid value for JPEG Quality. Must be a number between 1 and 100.", "warning")

            config['academic_year'] = request.form.get('academic_year', config.get('academic_year', ''))

            save_config(config)
            flash("General settings saved successfully.", "success")

        except Exception as e:
            logging.error(f"Error processing general configuration form: {e}")
            flash("An error occurred while saving general settings.", "error")

        action_taken = False
        if 'add_camera' in request.form and not action_taken:
            try:
                name = request.form.get('camera_name')
                ip = request.form.get('camera_ip')
                port = request.form.get('camera_port')
                username = request.form.get('camera_username')
                password = request.form.get('camera_password')
                rtsp_url = request.form.get('camera_rtsp_url')
                if name and ip and port and username is not None and password is not None and rtsp_url:
                    new_id = max([c.get('id', 0) for c in cameras if isinstance(c.get('id'), int)], default=0) + 1
                    cameras.append({
                        "id": new_id,
                        "name": name,
                        "ip": ip,
                        "port": int(port),
                        "username": username,
                        "password": password,
                        "rtsp_url": rtsp_url,
                        "disabled": False
                    })
                    save_cameras(cameras)
                    flash(Markup(f"Camera <b>{name}</b> added successfully."), "success")
                    action_taken = True
                else:
                    flash("All camera fields are required to add a new camera.", "danger")
            except ValueError:
                 flash("Invalid port number entered.", "danger")
            except Exception as e:
                 logging.error(f"Error adding camera: {e}")
                 flash("An error occurred while adding the camera.", "error")


        if 'remove_camera' in request.form and not action_taken:
            try:
                remove_id = int(request.form.get('remove_camera'))
                initial_length = len(cameras)
                cameras = [c for c in cameras if c['id'] != remove_id or remove_id == 0]
                if len(cameras) < initial_length:
                    save_cameras(cameras)
                    flash("Camera removed successfully.", "success")
                    action_taken = True
                elif remove_id == 0:
                    flash("Cannot remove the default PC Webcam.", "warning")
                else:
                    flash("Camera not found for removal.", "warning")
            except ValueError:
                 flash("Invalid camera ID for removal.", "danger")
            except Exception as e:
                 logging.error(f"Error removing camera: {e}")
                 flash("An error occurred while removing the camera.", "error")

        if 'toggle_camera' in request.form and not action_taken:
            try:
                toggle_id = int(request.form.get('toggle_camera'))
                toggled = False
                for cam in cameras:
                    if cam.get('id') == toggle_id:
                        if toggle_id == 0 and cam.get('disabled', False) == False:
                            enabled_cameras = sum(1 for c in cameras if not c.get('disabled', False))
                            if enabled_cameras <= 1:
                                flash("Cannot disable the last enabled camera (PC Webcam).", "warning")
                                toggled = True
                                break

                        cam['disabled'] = not cam.get('disabled', False)
                        save_cameras(cameras)
                        status = "disabled" if cam['disabled'] else "enabled"
                        flash(Markup(f"Camera <b>{cam['name']}</b> is now {status}."), "success" if status == "enabled" else "info")
                        toggled = True
                        action_taken = True
                        break
                if not toggled:
                     flash("Camera not found for enable/disable toggle.", "warning")

            except ValueError:
                 flash("Invalid camera ID for toggle.", "danger")
            except Exception as e:
                 logging.error(f"Error toggling camera status: {e}")
                 flash("An error occurred while toggling the camera status.", "error")

        return redirect(url_for('configure'))

    return render_template(
        'configure.html',
        stop_time_enabled=config.get('stop_time_enabled', False),
        current_stop_time=config.get('stop_time', '18:00'),
        unknown_face_timeout=config.get('unknown_face_timeout', 5),
        unknown_log_interval_seconds=config.get('unknown_log_interval_seconds', 60),
        academic_year=config.get('academic_year', ''),
        process_every_n_frames=config.get('process_every_n_frames', 10),
        live_feed_sleep=config.get('live_feed_sleep', 0.1),
        jpeg_quality=config.get('jpeg_quality', 60),
        cameras=cameras
    )

RECOGNITION_THRESHOLD = 0.36

def load_known_faces():
    global known_face_data, known_face_embeddings_list, known_face_names
    logging.info("Loading known faces from database...")
    known_face_data = {}
    temp_embeddings_list = []
    temp_names_list = []
    try:
        users_cursor = mongo.db.students.find({}, {"name": 1, "branch": 1, "role": 1, "face_embedding": 1, "_id": 1})
        count = 0
        skipped_count = 0
        for user_doc in users_cursor:
            user_id = user_doc.get('_id')
            user_name = user_doc.get('name')
            if 'face_embedding' in user_doc and user_doc['face_embedding'] and user_name:
                try:
                    embedding_np = np.array(user_doc['face_embedding']).astype(np.float32)
                    if embedding_np.shape == (128,):
                        known_face_data[user_name] = {
                            'embedding': embedding_np,
                            'branch': user_doc.get('branch'),
                            'role': user_doc.get('role', 'Unknown')
                        }
                        temp_embeddings_list.append(embedding_np)
                        temp_names_list.append(user_name)
                        count += 1
                    else:
                        logging.warning(f"Invalid embedding shape {embedding_np.shape} for user '{user_name}' (ID: {user_id}). Expected (128,). Skipping.")
                        skipped_count += 1
                except Exception as e:
                    logging.warning(f"Could not process embedding for user '{user_name}' (ID: {user_id}): {e}")
                    skipped_count += 1
            else:
                logging.warning(f"User data incomplete or missing face_embedding/name for user ID: {user_id}. Skipping.")
                skipped_count += 1

        known_face_embeddings_list = temp_embeddings_list
        known_face_names = temp_names_list
        if skipped_count > 0:
             logging.warning(f"Skipped loading {skipped_count} faces due to issues.")
        logging.info(f"Loaded {count} known faces (embeddings) into memory cache.")

    except Exception as e:
        logging.error(f"Error loading known faces from database: {e}", exc_info=True)


def detect_and_encode_face(image_np):
    if image_np is None or image_np.size == 0:
        logging.warning("detect_and_encode_face received an empty image.")
        return None, None

    if face_detector is None or face_recognizer is None:
        logging.error("Face Detector or Recognizer not initialized.")
        return None, None

    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif len(image_np.shape) != 3 or image_np.shape[2] != 3:
        logging.warning(f"Invalid image shape for detection: {image_np.shape}")
        return None, None

    height, width, _ = image_np.shape
    if width == 0 or height == 0:
         logging.warning("Input image has zero width or height.")
         return None, None

    # Error Fix: Restore setInputSize to tell detector the current frame size
    face_detector.setInputSize((width, height))

    try:
        status, faces = face_detector.detect(image_np)
    except cv2.error as e:
        # Log the specific error related to size mismatch if it occurs
        logging.error(f"Error during face detection: {e}")
        return None, None

    if faces is None or len(faces) == 0:
        return None, None

    best_face_index = 0
    max_area = 0
    if len(faces) > 1:
        areas = faces[:, 2] * faces[:, 3]
        best_face_index = np.argmax(areas)

    face_data = faces[best_face_index]
    box = face_data[0:4].astype(np.int32)
    confidence = face_data[-1]

    (x, y, w, h) = box
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)

    if w <= 0 or h <= 0:
        logging.warning(f"Invalid bounding box after boundary check: x={x}, y={y}, w={w}, h={h}")
        return None, (x, y, w, h)

    try:
        aligned_face = face_recognizer.alignCrop(image_np, face_data)

        if aligned_face is None or aligned_face.size == 0:
            logging.warning("Face alignment/cropping failed.")
            return None, (x, y, w, h)

        face_embedding = face_recognizer.feature(aligned_face)
        face_embedding_flat = face_embedding.flatten()
        return face_embedding_flat, (x, y, w, h)

    except cv2.error as e:
        logging.error(f"cv2 error during face alignment or feature extraction: {e}")
        return None, (x, y, w, h)
    except Exception as e:
        logging.error(f"Unexpected error during face encoding: {e}", exc_info=True)
        return None, (x, y, w, h)


load_known_faces()

@app.route('/refresh_faces', methods=['POST'])
def refresh_faces_route():
    if session.get('user_type') != 'admin':
        return jsonify({"message": "Unauthorized"}), 403
    try:
        initial_count = len(known_face_data)
        load_known_faces()
        new_count = len(known_face_data)
        message = f"Face cache refreshed. Loaded {new_count} faces (previously {initial_count})."
        flash(message, "success")
        return jsonify({"message": message}), 200
    except Exception as e:
        logging.error(f"Error during manual face cache refresh: {e}", exc_info=True)
        flash("Error refreshing face cache.", "error")
        return jsonify({"message": "Error refreshing face cache."}), 500

@app.route('/')
def index():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/live_feed')
def live_feed():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access to live feed.", "warning")
        return redirect(url_for('login'))
    cameras = load_cameras()
    enabled_cameras = [cam for cam in cameras if not cam.get('disabled', False)]
    return render_template('live_feed.html', cameras=enabled_cameras)


def generate_frames(source):
    logging.info(f"generate_frames started for source: {source}")
    logging.info(f"Known faces loaded: {len(known_face_names)}")
    config = load_config()

    stop_time_enabled = config.get('stop_time_enabled', False)
    stop_time_str = config.get('stop_time')
    process_every_n_frames = max(1, config.get('process_every_n_frames', 10))
    live_feed_sleep = max(0.0, config.get('live_feed_sleep', 0.1))
    jpeg_quality = max(1, min(100, config.get('jpeg_quality', 60)))
    unknown_log_interval_seconds = config.get('unknown_log_interval_seconds', 60)
    resize_width = config.get('resize_width', None)
    resize_height = config.get('resize_height', None)

    log_interval = timedelta(minutes=1)
    unknown_log_interval = timedelta(seconds=unknown_log_interval_seconds)
    logging.info(f"Using settings - Process every {process_every_n_frames} frames, Sleep: {live_feed_sleep}s, JPEG Quality: {jpeg_quality}")

    stop_time_obj = None
    india_tz = pytz.timezone('Asia/Kolkata')

    if stop_time_enabled and stop_time_str:
        try:
            stop_time_obj = datetime.strptime(stop_time_str, '%H:%M').time()
            logging.info(f"Live feed monitoring enabled to stop at {stop_time_str} (India Time)")
        except ValueError:
            logging.warning(f"Invalid stop_time format '{stop_time_str}' in config. Ignoring stop time.")
            stop_time_obj = None
    elif stop_time_enabled:
        logging.warning("Stop time is enabled in config, but no valid time is set.")

    cap = None
    try:
        if isinstance(source, str) and source.startswith("rtsp://"):
            logging.info(f"Attempting to open RTSP stream: {source}")
            if "?" in source: source += "&rtsp_transport=tcp"
            else: source += "?rtsp_transport=tcp"
            logging.info(f"Using RTSP URL with forced TCP: {source}")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            buffer_size = 32
            success_set_buffer = cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            if success_set_buffer: logging.info(f"Successfully set RTSP buffer size to {buffer_size}.")
            else: logging.warning("Failed to set buffer size for RTSP stream.")
        else:
            logging.info(f"Attempting to open webcam/other source: {source}")
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for backend in backends:
                cap = cv2.VideoCapture(source, backend)
                if cap.isOpened():
                    logging.info(f"Successfully opened source {source} using backend {backend}")
                    break
                else:
                     cap.release()
                     logging.warning(f"Failed to open source {source} with backend {backend}")
            if cap is None or not cap.isOpened():
                logging.error(f"Failed to open source {source} with any backend.")
                return

        if not cap.isOpened():
            logging.error(f"Cannot open video source: {source}")
            return

        frame_count = 0
        last_known_faces_log = {}
        last_unknown_log_time = None
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 30

        while True:
            if stop_time_enabled and stop_time_obj:
                current_time_india = datetime.now(india_tz).time()
                if current_time_india >= stop_time_obj:
                    logging.info(f"Stop time {stop_time_obj} reached. Stopping live feed.")
                    break

            try:
                grabbed = cap.grab()
                if not grabbed:
                     consecutive_errors += 1
                     logging.warning(f"cap.grab() failed for source {source}. Consecutive errors: {consecutive_errors}")
                     if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                         logging.error(f"Max consecutive grab errors ({MAX_CONSECUTIVE_ERRORS}) reached. Stopping generator.")
                         break
                     time.sleep(0.05)
                     continue

                if frame_count % process_every_n_frames == 0:
                    ret, frame = cap.retrieve()
                    # Resize frame if config values are set
                    if ret and frame is not None and resize_width and resize_height:
                        frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
                    if not ret or frame is None:
                        consecutive_errors += 1
                        logging.warning(f"cap.retrieve() failed after grab for source {source}. Consecutive errors: {consecutive_errors}")
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logging.error(f"Max consecutive retrieve errors ({MAX_CONSECUTIVE_ERRORS}) reached. Stopping generator.")
                            break
                        continue
                else:
                     frame_count += 1
                     continue

            except cv2.error as e:
                logging.error(f"cv2 error during frame grab/retrieve from source {source}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS: break
                time.sleep(0.1)
                continue

            consecutive_errors = 0

            if not hasattr(frame, "shape") or len(frame.shape) != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
                logging.warning(f"Invalid frame retrieved from source {source}: {type(frame)}, shape: {getattr(frame, 'shape', None)}. Skipping.")
                frame_count += 1
                continue

            processing_frame = frame.copy()
            detected_box = None
            status_text = "Processing..."
            status_color = (0, 255, 255)

            try:
                face_embedding, detected_box = detect_and_encode_face(processing_frame)

                if detected_box:
                    logging.debug("Face detected in frame.")
                    (x, y, w, h) = detected_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if face_embedding is not None:
                        match_found = False
                        min_distance = float('inf')

                        if known_face_embeddings_list:
                            distances = [cosine(face_embedding, known_emb) for known_emb in known_face_embeddings_list]
                            if distances:
                                min_distance_idx = np.argmin(distances)
                                min_distance = distances[min_distance_idx]

                                if min_distance < RECOGNITION_THRESHOLD:
                                    name = known_face_names[min_distance_idx]
                                    logging.debug(f"Match found: {name} (distance: {min_distance:.2f})")
                                    status_text = f"Known: {name} ({min_distance:.2f})"
                                    status_color = (0, 255, 0)
                                    match_found = True

                                    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
                                    log_entry = {
                                        'type': 'known_sighting',
                                        'name': name,
                                        'timestamp': now_utc,
                                        'status_at_log': 'Known'
                                    }
                                    try:
                                        # Fetch branch and academic_year from students collection
                                        student_doc = mongo.db.students.find_one({'name': name}, {"branch": 1, "academic_year": 1})
                                        if student_doc and student_doc.get("branch") and student_doc.get("academic_year"):
                                            branch = student_doc["branch"].lower().replace(".", "").replace(" ", "_")
                                            year = student_doc["academic_year"].lower().replace(".", "").replace(" ", "_")
                                            collection_name = f"{branch}_{year}_year"
                                            # Insert into the specific collection
                                            getattr(mongo.db, collection_name).insert_one(log_entry)
                                            logging.debug(f"Logged known sighting for {name} at {now_utc} in collection {collection_name}")
                                        else:
                                            # Fallback to default collection if details are missing
                                            mongo.db.seen_log.insert_one(log_entry)
                                            logging.warning(f"Branch/year missing for {name}, logged in default seen_log collection.")
                                    except Exception as log_e:
                                        logging.error(f"Failed to log known sighting for {name}: {log_e}")

                        if not match_found:
                            status_text = f"Unknown ({min_distance:.2f})" if min_distance != float('inf') else "Unknown"
                            status_color = (0, 0, 255)
                            now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
                            if not last_unknown_log_time or (now_utc - last_unknown_log_time) > unknown_log_interval:
                                face_img_b64 = None
                                try:
                                    face_region = processing_frame[y:y+h, x:x+w]
                                    if face_region.size > 0:
                                        _, buffer = cv2.imencode('.jpg', face_region, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                                        face_img_b64 = base64.b64encode(buffer).decode('utf-8')
                                except Exception as img_err: logging.error(f"Error encoding unknown face image: {img_err}")

                                log_entry = {'type': 'unknown_sighting', 'timestamp': now_utc, 'status_at_log': 'Unknown', 'face_image_base64': face_img_b64}
                                try:
                                    mongo.db.seen_log.insert_one(log_entry)
                                    logging.info(f"Logged unknown sighting at {now_utc}")
                                except Exception as log_e: logging.error(f"Failed to log unknown sighting: {log_e}")
                                last_unknown_log_time = now_utc
                    else:
                         status_text = "Processing Error"
                         status_color = (255, 0, 0)
                else:
                     status_text = "No Face Detected"
                     status_color = (255, 255, 255)

            except cv2.error as e:
                logging.error(f"cv2 error during face processing: {e}")
                status_text = "Detection/Encoding Error"
                status_color = (255, 0, 0)
            except Exception as e:
                logging.error(f"Unexpected error during face processing: {e}", exc_info=True)
                status_text = "Unexpected Processing Error"
                status_color = (255, 0, 0)

            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                if not ret:
                    logging.warning(f"cv2.imencode failed for source {source}. Skipping frame.")
                    frame_count += 1
                    continue
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except cv2.error as e:
                 logging.error(f"cv2 error during frame encoding for source {source}: {e}")
                 frame_count += 1
                 continue
            except Exception as e:
                 logging.error(f"Unexpected error during frame encoding/yielding for source {source}: {e}")
                 break

            if live_feed_sleep > 0:
                time.sleep(live_feed_sleep)

            frame_count += 1

    except (GeneratorExit, ConnectionResetError, BrokenPipeError) as e:
        logging.info(f"Client disconnected or connection lost for source {source}: {type(e).__name__}")
    except Exception as e:
        logging.error(f"Unexpected error in video stream generator for source {source}: {e}", exc_info=True)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            logging.info(f"Video source {source} released in finally block.")
        else:
            logging.info(f"Video source {source} was not open or already released.")
        logging.info(f"generate_frames exited for source: {source}")


@app.route('/video_feed')
def video_feed():
    if session.get('user_type') != 'admin':
        logging.warning("Unauthorized attempt to access video_feed endpoint.")
        return Response("Unauthorized", status=403)

    camera_id_str = request.args.get('camera_id')
    cameras = load_cameras()
    camera = None
    source_to_use = None

    logging.debug(f"Request for video_feed with camera_id: {camera_id_str}")

    if camera_id_str is not None:
        try:
            if camera_id_str == '0': camera_id = 0
            else: camera_id = int(camera_id_str)

            for cam in cameras:
                if (cam.get('id') == camera_id or (camera_id == 0 and str(cam.get('id')) == '0')) and not cam.get('disabled', False):
                    camera = cam
                    break

            if camera:
                source_to_use = camera.get('rtsp_url')
                if camera.get('id') == 0 and (source_to_use == 0 or str(source_to_use) == '0'):
                    source_to_use = 0
                elif isinstance(source_to_use, str) and source_to_use.isdigit():
                     try: source_to_use = int(source_to_use)
                     except ValueError: pass

                logging.info(f"Streaming from selected camera: {camera['name']} (ID: {camera_id}), Source: {source_to_use}")
            else:
                 logging.warning(f"Requested Camera ID {camera_id_str} not found or is disabled.")
                 return Response(f"Camera ID {camera_id_str} not found or is disabled", status=404)

        except ValueError:
             logging.warning(f"Invalid camera_id format provided: {camera_id_str}")
             return Response(f"Invalid camera_id format: {camera_id_str}", status=400)
    else:
        logging.info("No camera_id specified, searching for first enabled camera.")
        for cam in cameras:
            if not cam.get('disabled', False):
                camera = cam
                source_to_use = camera.get('rtsp_url')
                if camera.get('id') == 0 and (source_to_use == 0 or str(source_to_use) == '0'):
                     source_to_use = 0
                elif isinstance(source_to_use, str) and source_to_use.isdigit():
                     try: source_to_use = int(source_to_use)
                     except ValueError: pass
                logging.info(f"Using first enabled camera found: {camera['name']} (ID: {camera.get('id')}), Source: {source_to_use}")
                break

        if source_to_use is None:
            logging.error("Could not find any enabled cameras to stream.")
            return Response("No enabled cameras available", status=404)

    load_known_faces()
    logging.info(f"Starting generate_frames with source: {source_to_use}")
    return Response(generate_frames(source_to_use), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_type') != 'admin':
        flash("Only admins can register new users.", "warning")
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
            flash(f"Invalid or missing role. Must be one of: {', '.join(allowed_roles)}.", "error")
            return render_template('register.html'), 400

        if not name:
            flash("Name is required.", "error")
            return render_template('register.html'), 400

        if role == 'student':
            if not branch:
                flash("Branch is required for students.", "error")
                return render_template('register.html'), 400
            valid_academic_years = ['1st', '2nd', '3rd']
            if not academic_year or academic_year not in valid_academic_years:
                flash("Valid Academic Year (1st, 2nd, or 3rd) is required for students.", "error")
                return render_template('register.html'), 400
        elif role == 'staff':
            if not branch:
                flash("Branch/Department is required for staff.", "error")
                return render_template('register.html'), 400
            academic_year = None
        else:
             branch = None
             academic_year = None

        image = None
        image_source = "unknown"
        temp_file_path = None

        if photo_data and photo_data.startswith('data:image/'):
            try:
                header, encoded = photo_data.split(',', 1)
                image_bytes = io.BytesIO(base64.b64decode(encoded))
                nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None: raise ValueError("Decoded image is None")
                image_source = "webcam"
                logging.info("Processing image from webcam data.")
            except Exception as e:
                logging.error(f"Error decoding base64 image: {e}")
                flash("Invalid photo data from webcam.", "error")
                return render_template('register.html'), 400
        elif uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
            try:
                uploaded_file.save(temp_file_path)
                image = cv2.imread(temp_file_path)
                if image is None:
                    raise ValueError(f"cv2.imread returned None for {filename}. Check file format/integrity.")
                image_source = f"upload ({filename})"
                logging.info(f"Processing image from uploaded file: {filename}")
            except Exception as e:
                logging.error(f"Error saving or reading uploaded file '{filename}': {e}")
                flash(f"Could not process uploaded file: {e}", "error")
                if temp_file_path and os.path.exists(temp_file_path):
                     try: os.remove(temp_file_path)
                     except OSError as rm_err: logging.error(f"Error removing temporary file {temp_file_path}: {rm_err}")
                return render_template('register.html'), 400
        else:
            flash("No photo provided (either webcam capture or file upload is required).", "error")
            return render_template('register.html'), 400

        if image is None:
            flash("Failed to load image data.", "error")
            logging.error("Image data is None after loading attempts.")
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_err: logging.error(f"Error removing temporary file {temp_file_path}: {rm_err}")
            return render_template('register.html'), 500

        try:
            existing_user = mongo.db.students.find_one({'name': name})
            if existing_user:
                logging.warning(f"Registration attempt for existing user: {name}")
                flash(f"User '{name}' is already registered!", "warning")
                if temp_file_path and os.path.exists(temp_file_path):
                    try: os.remove(temp_file_path)
                    except OSError as rm_err: logging.error(f"Error removing temporary file {temp_file_path}: {rm_err}")
                return render_template('register.html'), 400
        except Exception as e:
            logging.error(f"Database error checking for existing user {name}: {e}")
            flash("Database error checking for existing user.", "error")
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_err: logging.error(f"Error removing temporary file {temp_file_path}: {rm_err}")
            return render_template('register.html'), 500


        logging.info(f"Processing registration for {name} from {image_source}.")
        face_embedding, box = detect_and_encode_face(image)

        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError as e: logging.error(f"Error removing upload file {temp_file_path} after processing: {e}")

        if face_embedding is None:
            message = "No face detected in the provided image. Please ensure the face is clear and well-lit."
            if box:
                message = "Face detected, but could not process it for registration. Try a clearer image or different angle."
            logging.warning(f"No face detected or embedding failed for registration image of {name}.")
            flash(message, "error")
            return render_template('register.html'), 400

        try:
            embedding_list = face_embedding.tolist()
            user_data = {
                'name': name,
                'role': role,
                'face_embedding': embedding_list,
                'registered_at': datetime.utcnow(),
                'branch': branch,
                'academic_year': academic_year
            }

            result = mongo.db.students.insert_one(user_data)
            logging.info(f"Successfully registered user: {name} (ID: {result.inserted_id}) with role: {role}, branch: {branch}, year: {academic_year}")
            flash(f"User '{name}' ({role.capitalize()}) registered successfully!", "success")

            load_known_faces()

            return redirect(url_for('view_students'))

        except Exception as e:
            logging.error(f"Database error during registration for {name}: {e}", exc_info=True)
            flash("Database error occurred during registration.", "error")
            return render_template('register.html'), 500

    return render_template('register.html')


@app.route('/view_students', methods=['GET'])
def view_students():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        search_query = request.args.get('search', '').strip()
        role_filter = request.args.get('role_filter', '')
        branch_filter = request.args.get('branch_filter', '')
        academic_year_filter = request.args.get('academic_year_filter', '')
        sort_by = request.args.get('sort_by', 'name')
        sort_order = request.args.get('sort_order', 'asc')
        india_tz = pytz.timezone('Asia/Kolkata')

        distinct_branches = []
        try:
            branch_query = {'branch': {'$nin': [None, '']}, 'role': {'$in': ['student', 'staff']}}
            distinct_branches = mongo.db.students.distinct('branch', branch_query)
            distinct_branches = sorted([b for b in distinct_branches if isinstance(b, str)])
        except Exception as db_err:
            logging.error(f"Error fetching distinct branches: {db_err}")
            flash("Error loading branch filter options.", "warning")

        query_conditions = []
        if search_query:
            regex = {'$regex': search_query, '$options': 'i'}
            query_conditions.append({'$or': [{'name': regex}, {'branch': regex}]})
        if role_filter:
            query_conditions.append({'role': role_filter})
        if branch_filter and (role_filter in ['student', 'staff'] or not role_filter):
            query_conditions.append({'branch': branch_filter})
        if academic_year_filter and (role_filter == 'student' or not role_filter):
             query_conditions.append({'academic_year': academic_year_filter})

        query = {'$and': query_conditions} if query_conditions else {}
        logging.debug(f"Executing student query: {query}")

        mongo_sort_order = 1 if sort_order == 'asc' else -1
        valid_sort_fields = ['name', 'role', 'registered_at', 'branch', 'academic_year']
        if sort_by not in valid_sort_fields: sort_by = 'name'

        students_cursor = mongo.db.students.find(query).sort(sort_by, mongo_sort_order)
        students_list = list(students_cursor)
        logging.info(f"Found {len(students_list)} students matching filters.")

        for student in students_list:
            reg_at_utc = student.get('registered_at')
            if isinstance(reg_at_utc, datetime):
                 reg_at_kolkata = reg_at_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                 student['registered_at_str'] = reg_at_kolkata.strftime('%Y-%m-%d %H:%M:%S %Z')
            else: student['registered_at_str'] = "N/A"
            student['_id_str'] = str(student['_id'])
            student['role_display'] = student.get('role', 'N/A').capitalize()
            student['branch_display'] = student.get('branch') or 'N/A'
            student['academic_year_display'] = student.get('academic_year') or 'N/A'

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
        logging.exception("Error fetching students list for view:")
        flash("Error loading student data. Please check logs.", "error")
        distinct_branches_on_error = []
        try:
            branch_query = {'branch': {'$nin': [None, '']}, 'role': {'$in': ['student', 'staff']}}
            distinct_branches_on_error = mongo.db.students.distinct('branch', branch_query)
            distinct_branches_on_error = sorted([b for b in distinct_branches_on_error if isinstance(b, str)])
        except: pass

        return render_template('view_students.html', students=[],
                               search_query=request.args.get('search', ''),
                               role_filter=request.args.get('role_filter', ''),
                               branch_filter=request.args.get('branch_filter', ''),
                               academic_year_filter=request.args.get('academic_year_filter', ''),
                               distinct_branches=distinct_branches_on_error,
                               sort_by=request.args.get('sort_by', 'name'),
                               sort_order=request.args.get('sort_order', 'asc')), 500


@app.route('/delete_student/<student_id>', methods=['POST'])
def delete_student(student_id):
    if session.get('user_type') != 'admin':
         flash("Unauthorized action.", "error")
         return redirect(url_for('view_students'))

    try:
        oid = ObjectId(student_id)
    except Exception:
        flash("Invalid student ID format.", "error")
        return redirect(url_for('view_students'))

    try:
        student_to_delete = mongo.db.students.find_one({'_id': oid}, {'name': 1})
        if not student_to_delete:
            flash("Student not found for deletion.", "error")
            return redirect(url_for('view_students')), 404

        student_name = student_to_delete.get('name', f'ID_{student_id}')

        result = mongo.db.students.delete_one({'_id': oid})

        if result.deleted_count > 0:
            logging.info(f"Successfully deleted student: {student_name} (ID: {student_id})")
            flash(f"Student '{student_name}' deleted successfully.", "success")
            load_known_faces()

            try:
                log_result = mongo.db.seen_log.delete_many({'name': student_name, 'type': 'known_sighting'})
                if log_result.deleted_count > 0:
                    logging.info(f"Removed {log_result.deleted_count} sighting logs for deleted student {student_name}.")
            except Exception as log_e:
                logging.error(f"Error removing sighting logs for {student_name}: {log_e}")
                flash("Student deleted, but failed to remove associated logs.", "warning")
        else:
            logging.warning(f"Deletion query for student {student_name} (ID: {student_id}) reported 0 deleted.")
            flash("Student found but could not be deleted.", "error")

    except Exception as e:
        logging.error(f"Error deleting student {student_id}: {e}", exc_info=True)
        flash("An error occurred while trying to delete the student.", "error")

    return redirect(url_for('view_students'))


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
            session.permanent = True
            app.permanent_session_lifetime = timedelta(hours=8)
            logging.info("Admin login successful.")
            flash("Admin login successful.", "success")
            return redirect(url_for('index'))
        else:
            logging.warning("Invalid admin password attempt.")
            flash("Invalid admin credentials.", "error")
            return render_template('login.html'), 401

    return render_template('login.html')


@app.route('/logout')
def logout():
    user_name = session.get('user_name', 'User')
    session.clear()
    logging.info(f"User '{user_name}' logged out.")
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('login'))


@app.before_request
def restrict_access():
    if request.endpoint in ['login', 'static', 'logout']:
        return None

    if 'user_type' not in session:
        flash("Please log in to access this page.", "info")
        return redirect(url_for('login'))

    if session.get('user_type') == 'admin':
        return None

    flash("Admin access required. Please log in.", "warning")
    return redirect(url_for('login'))


@app.route('/seen_log')
def seen_log_view():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))
    try:
        log_records_cursor = mongo.db.seen_log.find().sort('timestamp', -1)
        log_records = list(log_records_cursor)
        logging.debug(f"Fetched {len(log_records)} seen log records.")

        grouped_logs = defaultdict(list)
        india_tz = pytz.timezone('Asia/Kolkata')

        for record in log_records:
            ts_utc = record.get('timestamp')
            if isinstance(ts_utc, datetime):
                ts_kolkata = ts_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                record['timestamp_str'] = ts_kolkata.strftime('%Y-%m-%d %H:%M:%S %Z')
            else: record['timestamp_str'] = "Invalid Date"

            record_type = record.get('type')
            group_key = "Other"
            if record_type == 'known_sighting':
                group_key = record.get('name', 'Known (Error)')
                if not record.get('name'): record['name'] = 'Known (Error)'
            elif record_type in ['unknown_sighting', 'processing_error_sighting']:
                group_key = 'Unknown'
                if 'face_image_base64' not in record: record['face_image_base64'] = None

            grouped_logs[group_key].append(record)

        def sort_key(item):
            key = item[0]
            if key == 'Unknown': return (1, key)
            if key == 'Other': return (2, key)
            if key == 'Known (Error)': return (0, '~')
            return (0, key)

        sorted_grouped_logs = dict(sorted(grouped_logs.items(), key=sort_key))

        return render_template('seen_log.html', grouped_logs=sorted_grouped_logs)

    except Exception as e:
        logging.error(f"Error fetching or grouping seen log records: {e}", exc_info=True)
        flash("Error loading seen log data.", "error")
        return render_template('seen_log.html', grouped_logs={}), 500


@app.route('/unknown_captures')
def unknown_captures():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')
        time_filter = {}
        india_tz = pytz.timezone('Asia/Kolkata')
        start_time_value = ''
        end_time_value = ''

        if start_time_str:
            try:
                start_dt_local = datetime.fromisoformat(start_time_str)
                start_dt_aware_local = india_tz.localize(start_dt_local)
                time_filter['$gte'] = start_dt_aware_local.astimezone(pytz.utc)
                start_time_value = start_time_str
            except (ValueError, TypeError) as e:
                flash(f"Invalid start time format ignored: {e}", "warning")
        if end_time_str:
            try:
                end_dt_local = datetime.fromisoformat(end_time_str)
                end_dt_aware_local = india_tz.localize(end_dt_local)
                time_filter['$lte'] = end_dt_aware_local.astimezone(pytz.utc)
                end_time_value = end_time_str
            except (ValueError, TypeError) as e:
                flash(f"Invalid end time format ignored: {e}", "warning")

        query = {"type": {"$in": ["unknown_sighting", "processing_error_sighting"]}}
        if time_filter:
            query['timestamp'] = time_filter
        logging.debug(f"Unknown captures query: {query}")

        log_records_cursor = mongo.db.seen_log.find(query).sort('timestamp', -1)
        log_records = list(log_records_cursor)
        logging.debug(f"Fetched {len(log_records)} unknown capture records.")

        grouped_unknowns = defaultdict(list)
        for record in log_records:
            ts_utc = record.get('timestamp')
            if isinstance(ts_utc, datetime):
                ts_kolkata = ts_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                record['timestamp_str'] = ts_kolkata.strftime('%Y-%m-%d %H:%M:%S %Z')
                date_str = ts_kolkata.strftime('%Y-%m-%d')
            else:
                record['timestamp_str'] = "Invalid Date"
                date_str = "Unknown Date"
            if 'face_image_base64' not in record: record['face_image_base64'] = None
            grouped_unknowns[date_str].append(record)

        grouped_unknowns_dict = dict(grouped_unknowns)

        return render_template('unknown_captures.html',
                                grouped_unknowns=grouped_unknowns_dict,
                                start_time_value=start_time_value.replace(' ', 'T'),
                                end_time_value=end_time_value.replace(' ', 'T'))

    except Exception as e:
        logging.error(f"Error fetching or grouping unknown captures: {e}", exc_info=True)
        flash("Error loading unknown captures data.", "error")
        return render_template('unknown_captures.html', grouped_unknowns={}), 500


@app.route('/reset_seen_log', methods=['POST'])
def reset_seen_log():
    if session.get('user_type') != 'admin':
        logging.warning("Unauthorized attempt to reset seen log.")
        flash("Unauthorized action.", "error")
        return redirect(url_for('login'))

    try:
        result = mongo.db.seen_log.delete_many({})
        deleted_count = result.deleted_count
        logging.info(f"Seen log reset requested by admin. {deleted_count} records deleted.")
        flash(f"Seen log reset successfully. {deleted_count} records were deleted.", "success")
    except Exception as e:
        logging.error(f"Error resetting seen log: {e}", exc_info=True)
        flash("An error occurred while resetting the seen log.", "error")

    return redirect(url_for('seen_log_view'))


@app.route('/attendance', methods=['GET', 'POST'])
def attendance_page():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    india_tz = pytz.timezone('Asia/Kolkata')
    now_utc = datetime.utcnow()
    now_india = now_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
    today_date_str = now_india.strftime('%Y-%m-%d')
    # Default values for GET
    attendance_records = []
    branch_filter = ''
    academic_year_filter = ''
    selected_date = today_date_str

    if request.method == 'POST':
        branch_filter = request.form.get('branch_filter', '')
        academic_year_filter = request.form.get('academic_year_filter', '')
        selected_date = request.form.get('attendance_date', today_date_str)
        try:
            # Parse selected_date (YYYY-MM-DD) to date object
            try:
                selected_date_obj = datetime.strptime(selected_date, "%Y-%m-%d").date()
            except Exception:
                selected_date_obj = now_india.date()
            start_of_day_utc = india_tz.localize(datetime.combine(selected_date_obj, time.min)).astimezone(pytz.utc)
            end_of_day_utc = india_tz.localize(datetime.combine(selected_date_obj, time.max)).astimezone(pytz.utc)
            logging.info(f"Attendance Query: Fetching logs between {start_of_day_utc} (UTC) and {end_of_day_utc} (UTC)")

            # Determine which collection to query based on filters
            if branch_filter and academic_year_filter:
                branch = branch_filter.lower().replace(".", "").replace(" ", "_")
                year = academic_year_filter.lower().replace(".", "").replace(" ", "_")
                collection_name = f"{branch}_{year}_year"
                collection = getattr(mongo.db, collection_name)
            else:
                collection = mongo.db.seen_log

            log_records_cursor = collection.find({
                'type': 'known_sighting',
                'timestamp': {'$gte': start_of_day_utc, '$lte': end_of_day_utc}
            }).sort('timestamp', 1)

            raw_log_count = collection.count_documents({
                 'type': 'known_sighting',
                 'timestamp': {'$gte': start_of_day_utc, '$lte': end_of_day_utc}
            })
            logging.info(f"Attendance Query: Found {raw_log_count} raw known_sighting logs for today.")

            user_details = {}
            try:
                known_users = mongo.db.students.find({}, {"name": 1, "branch": 1, "academic_year": 1, "_id": 0})
                for user in known_users:
                    user_details[user['name']] = {
                        'branch': user.get('branch') or 'N/A',
                        'academic_year': user.get('academic_year') or 'N/A'
                    }
            except Exception as db_err:
                logging.error(f"Error prefetching user details: {db_err}")

            attendance_data = defaultdict(list)
            for record in log_records_cursor:
                name = record.get('name')
                timestamp_utc = record.get('timestamp')
                if name and isinstance(timestamp_utc, datetime):
                     attendance_data[name].append(timestamp_utc)

            logging.info(f"Attendance Processing: Grouped logs into {len(attendance_data)} users.")

            logging.debug(f"Attendance Filters - Branch: '{branch_filter}', Academic Year: '{academic_year_filter}'")

            final_attendance = []
            for name, timestamps_utc in attendance_data.items():
                if timestamps_utc:
                    arriving_time_utc = timestamps_utc[0]
                    leaving_time_utc = timestamps_utc[-1]
                    arriving_time_kolkata = arriving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                    leaving_time_kolkata = leaving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                    details = user_details.get(name, {'branch': 'N/A', 'academic_year': 'N/A'})

                    branch_matches = (not branch_filter) or (details['branch'] == branch_filter)
                    year_matches = (not academic_year_filter) or (details['academic_year'] == academic_year_filter)

                    if branch_matches and year_matches:
                        final_attendance.append({
                            'name': name,
                            'branch': details['branch'],
                            'academic_year': details['academic_year'],
                            'arriving_time': arriving_time_kolkata.strftime('%H:%M:%S'),
                            'leaving_time': leaving_time_kolkata.strftime('%H:%M:%S'),
                            'date': arriving_time_kolkata.strftime('%Y-%m-%d')
                        })
                    else:
                        logging.debug(f"Filtered out '{name}'. Branch Match: {branch_matches} (Filter: '{branch_filter}', Record: '{details['branch']}'). Year Match: {year_matches} (Filter: '{academic_year_filter}', Record: '{details['academic_year']}')")

            logging.info(f"Processed attendance for {len(attendance_data)} individuals, {len(final_attendance)} matched filters.")
            final_attendance.sort(key=lambda x: x['name'])
            attendance_records = final_attendance

        except Exception as e:
            logging.error(f"Error generating attendance page: {e}", exc_info=True)
            flash("Error loading attendance data.", "error")
            attendance_records = []

    return render_template('attendance.html',
                           attendance_records=attendance_records,
                           branch_filter=branch_filter if branch_filter is not None else '',
                           academic_year_filter=academic_year_filter if academic_year_filter is not None else '',
                           today_date=today_date_str,
                           selected_date=selected_date)


@app.route('/export_attendance_excel')
def export_attendance_excel():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now_utc = datetime.utcnow()
        now_india = now_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
        current_date_str = now_india.strftime('%Y-%m-%d')
        start_of_day_utc = india_tz.localize(datetime.combine(now_india.date(), time.min)).astimezone(pytz.utc)
        end_of_day_utc = india_tz.localize(datetime.combine(now_india.date(), time.max)).astimezone(pytz.utc)

        log_records_cursor = mongo.db.seen_log.find({
            'type': 'known_sighting',
            'timestamp': {'$gte': start_of_day_utc, '$lte': end_of_day_utc}
        }).sort('timestamp', 1)

        user_details = {}
        known_users = mongo.db.students.find({}, {"name": 1, "branch": 1, "academic_year": 1, "_id": 0})
        for user in known_users:
            user_details[user['name']] = {'branch': user.get('branch') or 'N/A', 'academic_year': user.get('academic_year') or 'N/A'}

        attendance_data = defaultdict(list)
        for record in log_records_cursor:
            name = record.get('name')
            timestamp_utc = record.get('timestamp')
            if name and isinstance(timestamp_utc, datetime): attendance_data[name].append(timestamp_utc)

        branch_filter = request.args.get('branch_filter', '')
        academic_year_filter = request.args.get('academic_year_filter', '')
        logging.info(f"Exporting Excel with filters - Branch: '{branch_filter}', Year: '{academic_year_filter}'")

        final_attendance = []
        for name, timestamps_utc in attendance_data.items():
            if timestamps_utc:
                arriving_time_utc = timestamps_utc[0]
                leaving_time_utc = timestamps_utc[-1]
                arriving_time_kolkata = arriving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                leaving_time_kolkata = leaving_time_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                details = user_details.get(name, {'branch': 'N/A', 'academic_year': 'N/A'})

                branch_matches = (not branch_filter) or (details['branch'] == branch_filter)
                year_matches = (not academic_year_filter) or (details['academic_year'] == academic_year_filter)

                if branch_matches and year_matches:
                    final_attendance.append({
                        'name': name, 'branch': details['branch'], 'academic_year': details['academic_year'],
                        'arriving_time': arriving_time_kolkata.strftime('%H:%M:%S'),
                        'leaving_time': leaving_time_kolkata.strftime('%H:%M:%S'),
                        'date': arriving_time_kolkata.strftime('%Y-%m-%d')
                    })
        final_attendance.sort(key=lambda x: x['name'])

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet_title = f"Attendance {current_date_str}"
        if branch_filter: sheet_title += f"_{branch_filter.replace('.', '')}"
        if academic_year_filter: sheet_title += f"_{academic_year_filter}"
        sheet.title = sheet_title[:30]

        headers = ["Name", "Branch", "Academic Year", "Arriving Time", "Leaving Time", "Date"]
        sheet.append(headers)
        for record in final_attendance:
            sheet.append([record['name'], record['branch'], record['academic_year'],
                          record['arriving_time'], record['leaving_time'], record['date']])
        for col in sheet.columns:
             max_length = 0
             column_letter = col[0].column_letter
             for cell in col:
                 try:
                     if len(str(cell.value)) > max_length: max_length = len(cell.value)
                 except: pass
             adjusted_width = (max_length + 2)
             sheet.column_dimensions[column_letter].width = adjusted_width
 
        excel_stream = BytesIO()
        workbook.save(excel_stream)
        excel_stream.seek(0)
        logging.info(f"Generated Excel export for {len(final_attendance)} records.")

        filename_base = f"attendance_{current_date_str}"
        if branch_filter: filename_base += f"_{branch_filter.replace('.', '')}"
        if academic_year_filter: filename_base += f"_{academic_year_filter}"
        filename = f"{filename_base}.xlsx"

        return Response(
            excel_stream,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment;filename="{filename}"'}
        )

    except Exception as e:
        logging.error(f"Error generating Excel export: {e}", exc_info=True)
        flash("Error generating Excel export.", "error")
        return redirect(url_for('attendance_page',
                                branch_filter=request.args.get('branch_filter',''),
                                academic_year_filter=request.args.get('academic_year_filter','')))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)