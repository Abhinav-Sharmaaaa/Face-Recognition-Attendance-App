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
        "known_log_interval_seconds": 60, # Default known log interval
        "academic_year": "",
        "process_every_n_frames": 10, # Default frame skip
        "live_feed_sleep": 0.0,        # Default sleep time (0 means no artificial delay)
        "jpeg_quality": 60,           # Default JPEG quality
        "resize_width": 640,          # Default resize width
        "resize_height": 360          # Default resize height
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                # Ensure all keys exist, merging defaults with loaded values
                config = default_config.copy()
                config.update(loaded_config)
                # Validate numeric types after loading
                for key in ["unknown_face_timeout", "unknown_log_interval_seconds", "known_log_interval_seconds", "process_every_n_frames", "jpeg_quality", "resize_width", "resize_height"]:
                    if key in config and not isinstance(config[key], (int, float)):
                        logging.warning(f"Config key '{key}' has non-numeric value '{config[key]}'. Using default: {default_config[key]}")
                        config[key] = default_config[key]
                if "live_feed_sleep" in config and not isinstance(config["live_feed_sleep"], (int, float)):
                     logging.warning(f"Config key 'live_feed_sleep' has non-numeric value '{config['live_feed_sleep']}'. Using default: {default_config['live_feed_sleep']}")
                     config["live_feed_sleep"] = default_config["live_feed_sleep"]

                return config
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
                known_interval = int(request.form.get('known_log_interval_seconds', config.get('known_log_interval_seconds', 60)))
                config['known_log_interval_seconds'] = max(1, known_interval)
            except (ValueError, TypeError):
                flash("Invalid value for Known Log Interval. Must be a number.", "warning")

            try:
                n_frames = int(request.form.get('process_every_n_frames', config.get('process_every_n_frames', 10)))
                config['process_every_n_frames'] = max(1, n_frames)
            except (ValueError, TypeError):
                flash("Invalid value for Frame Skip. Must be a number.", "warning")

            try:
                feed_sleep = float(request.form.get('live_feed_sleep', config.get('live_feed_sleep', 0.0)))
                config['live_feed_sleep'] = max(0.0, feed_sleep)
            except (ValueError, TypeError):
                flash("Invalid value for Live Feed Sleep. Must be a number (e.g., 0.1).", "warning")

            try:
                quality = int(request.form.get('jpeg_quality', config.get('jpeg_quality', 60)))
                config['jpeg_quality'] = max(1, min(100, quality))
            except (ValueError, TypeError):
                flash("Invalid value for JPEG Quality. Must be a number between 1 and 100.", "warning")

            # Add resize options
            try:
                resize_w = int(request.form.get('resize_width', config.get('resize_width', 640)))
                config['resize_width'] = max(160, resize_w) # Set a minimum practical width
            except (ValueError, TypeError):
                 flash("Invalid value for Resize Width. Must be a number.", "warning")

            try:
                resize_h = int(request.form.get('resize_height', config.get('resize_height', 360)))
                config['resize_height'] = max(120, resize_h) # Set a minimum practical height
            except (ValueError, TypeError):
                 flash("Invalid value for Resize Height. Must be a number.", "warning")


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

    # Pass resize config to template
    return render_template(
        'configure.html',
        stop_time_enabled=config.get('stop_time_enabled', False),
        current_stop_time=config.get('stop_time', '18:00'),
        unknown_face_timeout=config.get('unknown_face_timeout', 5),
        unknown_log_interval_seconds=config.get('unknown_log_interval_seconds', 60),
        known_log_interval_seconds=config.get('known_log_interval_seconds', 60),
        academic_year=config.get('academic_year', ''),
        process_every_n_frames=config.get('process_every_n_frames', 10),
        live_feed_sleep=config.get('live_feed_sleep', 0.0),
        jpeg_quality=config.get('jpeg_quality', 60),
        resize_width=config.get('resize_width', 640),      # Pass resize width
        resize_height=config.get('resize_height', 360),    # Pass resize height
        cameras=cameras
    )

RECOGNITION_THRESHOLD = 0.50

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


# -*- coding: utf-8 -*-
# Replace the existing generate_frames function in your app.py with this one.
# Imports and other parts of the file remain the same.

# -*- coding: utf-8 -*-
# Replace the existing generate_frames function in your app.py with this one.
# Ensure RECOGNITION_THRESHOLD is set appropriately near the top of your file.
# Imports and other parts of the file remain the same.

def generate_frames(source):
    logging.info(f"generate_frames started for source: {source}")
    logging.info(f"Known faces loaded: {len(known_face_names)}")
    config = load_config() # Load latest config at start of stream

    stop_time_enabled = config.get('stop_time_enabled', False)
    stop_time_str = config.get('stop_time')
    process_every_n_frames = max(1, config.get('process_every_n_frames', 1))
    live_feed_sleep = max(0.0, config.get('live_feed_sleep', 0.0))
    jpeg_quality = max(1, min(100, config.get('jpeg_quality', 60)))
    unknown_log_interval_seconds = config.get('unknown_log_interval_seconds', 60)
    known_log_interval_seconds = config.get('known_log_interval_seconds', 60) # Load known log interval
    resize_width = config.get('resize_width', None)
    resize_height = config.get('resize_height', None)

    unknown_log_interval = timedelta(seconds=unknown_log_interval_seconds)
    known_log_interval = timedelta(seconds=known_log_interval_seconds) # Create timedelta for known interval

    logging.info(f"Using settings - Process every {process_every_n_frames} frames, Sleep: {live_feed_sleep}s, JPEG Quality: {jpeg_quality}, Resize: {resize_width}x{resize_height}")
    logging.info(f"Log Intervals - Unknown: {unknown_log_interval_seconds}s, Known: {known_log_interval_seconds}s")
    logging.info(f"Using Recognition Threshold: {RECOGNITION_THRESHOLD}") # Log the threshold being used

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
        # --- Camera opening logic (copied from your latest code) ---
        if isinstance(source, str) and source.startswith("rtsp://"):
            logging.info(f"Attempting to open RTSP stream: {source}")
            parsed_url = urllib.parse.urlparse(source)
            query = urllib.parse.parse_qs(parsed_url.query)
            query['rtsp_transport'] = ['tcp'] # Force TCP
            updated_query = urllib.parse.urlencode(query, doseq=True)
            source = urllib.parse.urlunparse(parsed_url._replace(query=updated_query))

            logging.info(f"Using RTSP URL with forced TCP: {source}")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            buffer_size = 3
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
                logging.error(f"Exiting generator: Failed to open source {source} with any backend.")
                return

        if not cap.isOpened():
            logging.error(f"Cannot open video source: {source}")
            logging.error(f"Exiting generator: Cannot open video source: {source}")
            return
        # --- End Camera opening logic ---

        frame_count = 0
        last_known_faces_log = {} # Stores last SUCCESSFUL log time for each known person
        last_unknown_log_time = None
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 30
        MAX_RECONNECT_ATTEMPTS = 5
        reconnect_attempts = 0

        while True:
            if stop_time_enabled and stop_time_obj:
                current_time_india = datetime.now(india_tz).time()
                if current_time_india >= stop_time_obj:
                    logging.info(f"Stop time {stop_time_obj} reached. Stopping live feed.")
                    logging.info(f"Exiting generator: Stop time {stop_time_obj} reached.")
                    break

            frame = None
            if frame_count % process_every_n_frames == 0:
                try:
                    # --- Frame reading and error handling (copied from your latest code) ---
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consecutive_errors += 1
                        logging.warning(f"cap.read() failed for source {source}. Consecutive errors: {consecutive_errors}")
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logging.error(f"Max consecutive read errors ({MAX_CONSECUTIVE_ERRORS}) reached. Attempting to reconnect stream.")
                            cap.release()
                            reconnect_attempts += 1
                            if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                                logging.error(f"Max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached. Stopping generator.")
                                break
                            pytime.sleep(2)
                            if isinstance(source, str) and source.startswith("rtsp://"):
                                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                            else:
                                cap = cv2.VideoCapture(source)

                            if not cap.isOpened():
                                logging.error(f"Reconnection failed for source {source}.")
                                pytime.sleep(5)
                                continue
                            else:
                                logging.info(f"Reconnected to stream: {source}")
                                consecutive_errors = 0
                        else:
                             pytime.sleep(0.05)
                        frame_count += 1
                        continue
                    else:
                        consecutive_errors = 0 # Reset error count on success
                    # --- End Frame reading and error handling ---

                except cv2.error as e:
                    logging.error(f"cv2 error during frame read from source {source}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logging.error(f"Max consecutive cv2 errors ({MAX_CONSECUTIVE_ERRORS}) reached. Stopping generator.")
                        break
                    pytime.sleep(0.1)
                    frame_count += 1
                    continue

                # --- Frame processing starts here ---
                if frame is not None and resize_width and resize_height:
                    frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

                if not hasattr(frame, "shape") or len(frame.shape) != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    logging.warning(f"Invalid frame retrieved from source {source}: {type(frame)}, shape: {getattr(frame, 'shape', None)}. Skipping.")
                    frame_count += 1
                    continue

                processing_frame = frame.copy()
                detected_box = None
                status_text = "Processing..."
                status_color = (0, 255, 255) # Yellow

                try:
                    face_embedding, detected_box = detect_and_encode_face(processing_frame)

                    if detected_box:
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

                                    # <<< CORE LOGIC FOR KNOWN FACE >>>
                                    if min_distance < RECOGNITION_THRESHOLD:
                                        name = known_face_names[min_distance_idx]
                                        status_text = f"Known: {name} ({min_distance:.2f})"
                                        status_color = (0, 255, 0) # Green
                                        match_found = True

                                        now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
                                        last_log_time = last_known_faces_log.get(name)

                                        # Check if enough time has passed since the last SUCCESSFUL log for this KNOWN person
                                        if not last_log_time or (now_utc - last_log_time) > known_log_interval:
                                            # * Prepare log entry BEFORE trying to insert *
                                            log_entry = {
                                                'type': 'known_sighting',
                                                'name': name,
                                                'timestamp': now_utc,
                                                'status_at_log': 'Known'
                                            }
                                            # <<< MODIFICATION START (Applied from previous response) >>>
                                            # Always log known sightings to the main seen_log collection
                                            collection_to_log = mongo.db.seen_log
                                            target_collection_name = "seen_log"
                                            logging.debug(f"Attempting to log known sighting for {name} to collection: {target_collection_name}")
                                            # <<< MODIFICATION END >>>

                                            try:
                                                # Attempt insertion
                                                insert_result = collection_to_log.insert_one(log_entry)

                                                # *** CRITICAL FIX: Update last log time ONLY on successful insert ***
                                                if insert_result.acknowledged:
                                                    last_known_faces_log[name] = now_utc # UPDATE TIMESTAMP *AFTER* SUCCESSFUL INSERT
                                                    logging.info(f"Successfully logged known sighting for {name} to {target_collection_name} at {now_utc} (Interval: {known_log_interval_seconds}s)") # Use logging.info
                                                else:
                                                     # Don't update last_known_faces_log here
                                                     logging.warning(f"Database insertion for known sighting of {name} to {target_collection_name} was not acknowledged.")

                                            except Exception as log_e:
                                                # Log the error, but crucially, DO NOT update last_known_faces_log[name] here
                                                logging.error(f"Failed to log known sighting for {name} to {collection_to_log.name}: {log_e}", exc_info=True) # Add exc_info
                                        # else: # Log skipped due to interval (Optional debug logging)
                                            # logging.debug(f"Skipping log for known {name}: Interval not passed.")
                                    # <<< END CORE LOGIC FOR KNOWN FACE >>>


                            # <<< LOGIC FOR UNKNOWN FACE >>>
                            if not match_found:
                                status_text = f"Unknown ({min_distance:.2f})" if min_distance != float('inf') else "Unknown"
                                status_color = (0, 0, 255) # Red
                                now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)

                                # Check if enough time has passed since the last UNKNOWN log
                                if not last_unknown_log_time or (now_utc - last_unknown_log_time) > unknown_log_interval:
                                    face_img_b64 = None
                                    try:
                                        face_region = processing_frame[y:y+h, x:x+w]
                                        if face_region.size > 0:
                                            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75] # Quality for unknown face log
                                            _, buffer = cv2.imencode('.jpg', face_region, encode_params)
                                            face_img_b64 = base64.b64encode(buffer).decode('utf-8')
                                    except Exception as img_err: logging.error(f"Error encoding unknown face image: {img_err}")

                                    log_entry = {'type': 'unknown_sighting', 'timestamp': now_utc, 'status_at_log': 'Unknown', 'face_image_base64': face_img_b64}
                                    try:
                                        # Attempt insertion for unknown
                                        unknown_insert_result = mongo.db.seen_log.insert_one(log_entry)
                                        # Update time only if successful
                                        if unknown_insert_result.acknowledged:
                                            last_unknown_log_time = now_utc # UPDATE TIMESTAMP *AFTER* SUCCESSFUL INSERT
                                            logging.info(f"Logged unknown sighting at {now_utc} (Interval: {unknown_log_interval_seconds}s)")
                                        else:
                                            logging.warning("Database insertion for unknown sighting was not acknowledged.")
                                    except Exception as log_e:
                                        # Don't update last_unknown_log_time here
                                        logging.error(f"Failed to log unknown sighting: {log_e}", exc_info=True)
                                # else: # Log skipped due to interval (Optional debug logging)
                                    # logging.debug("Skipping log for unknown face: Interval not passed.")
                            # <<< END LOGIC FOR UNKNOWN FACE >>>

                        else: # face_embedding is None
                            status_text = "Encoding Error"
                            status_color = (255, 0, 0) # Blue
                    else: # detected_box is None
                        status_text = "No Face Detected"
                        status_color = (255, 255, 255) # White

                # --- Error Handling for face processing ---
                except cv2.error as e:
                    logging.error(f"cv2 error during face processing: {e}")
                    status_text = "Detection/Encoding Error"
                    status_color = (255, 0, 0)
                except Exception as e:
                    logging.error(f"Unexpected error during face processing: {e}", exc_info=True)
                    status_text = "Unexpected Processing Error"
                    status_color = (255, 0, 0)
                # --- End Error Handling ---

                # Draw status text on the frame
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # --- Encode and yield the frame ---
                try:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                    if not ret:
                        logging.warning(f"cv2.imencode failed for source {source}. Skipping frame.")
                    else:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except cv2.error as e:
                     logging.error(f"cv2 error during frame encoding for source {source}: {e}")
                except Exception as e:
                     logging.error(f"Unexpected error during frame encoding/yielding for source {source}: {e}")
                     break # Exit loop on yield error
                # --- End Encode and yield ---

            # --- Loop control ---
            if live_feed_sleep > 0:
                 pytime.sleep(live_feed_sleep)

            frame_count += 1 # Increment frame count for frame skipping logic
            # --- End Loop control ---

    # --- Generator Exit/Error Handling ---
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
    # --- End Generator Exit/Error Handling ---

# --- Keep the rest of your app.py file as it is ---

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
            # Handle camera_id '0' explicitly for webcam
            if camera_id_str == '0': camera_id = 0
            else: camera_id = int(camera_id_str)

            for cam in cameras:
                # Match integer ID or string '0' for webcam
                if (cam.get('id') == camera_id or (camera_id == 0 and str(cam.get('id')) == '0')) and not cam.get('disabled', False):
                    camera = cam
                    break

            if camera:
                source_to_use = camera.get('rtsp_url')
                # Explicitly check for webcam source (0 or '0')
                if camera.get('id') == 0 and (source_to_use == 0 or str(source_to_use) == '0'):
                    source_to_use = 0 # Use integer 0 for webcam
                elif isinstance(source_to_use, str) and source_to_use.isdigit():
                     try: source_to_use = int(source_to_use)
                     except ValueError: pass # Keep as string if not purely digits

                logging.info(f"Streaming from selected camera: {camera['name']} (ID: {camera_id}), Source: {source_to_use} (Type: {type(source_to_use)})")
            else:
                 logging.warning(f"Requested Camera ID {camera_id_str} not found or is disabled.")
                 return Response(f"Camera ID {camera_id_str} not found or is disabled", status=404)

        except ValueError:
             logging.warning(f"Invalid camera_id format provided: {camera_id_str}")
             return Response(f"Invalid camera_id format: {camera_id_str}", status=400)
    else:
        # Default to first enabled camera if no ID specified
        logging.info("No camera_id specified, searching for first enabled camera.")
        for cam in cameras:
            if not cam.get('disabled', False):
                camera = cam
                source_to_use = camera.get('rtsp_url')
                if camera.get('id') == 0 and (source_to_use == 0 or str(source_to_use) == '0'):
                     source_to_use = 0 # Use integer 0 for webcam
                elif isinstance(source_to_use, str) and source_to_use.isdigit():
                     try: source_to_use = int(source_to_use)
                     except ValueError: pass
                logging.info(f"Using first enabled camera found: {camera['name']} (ID: {camera.get('id')}), Source: {source_to_use} (Type: {type(source_to_use)})")
                break

        if source_to_use is None: # Check if a source was actually found
            logging.error("Could not find any enabled cameras to stream.")
            return Response("No enabled cameras available", status=404)

    load_known_faces() # Ensure faces are loaded before starting generator
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

        # --- Role specific field validation ---
        if role == 'student':
            if not branch:
                flash("Branch is required for students.", "error")
                return render_template('register.html'), 400
            valid_academic_years = ['1st', '2nd', '3rd']
            if not academic_year or academic_year not in valid_academic_years:
                flash("Valid Academic Year (1st, 2nd, or 3rd) is required for students.", "error")
                return render_template('register.html'), 400
        elif role == 'staff':
            # Branch/Department is required for staff
            if not branch:
                flash("Branch/Department is required for staff.", "error")
                return render_template('register.html'), 400
            # Academic year is not applicable for staff
            academic_year = None # Explicitly set to None
        else: # Others
             # Branch and Academic year are not applicable
             branch = None
             academic_year = None
        # --- End Role specific field validation ---

        image = None
        image_source = "unknown"
        temp_file_path = None

        # Prioritize webcam capture if available
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
        # Fallback to uploaded file if no webcam data or webcam failed
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

        # Ensure image data is valid before proceeding
        if image is None:
            flash("Failed to load image data.", "error")
            logging.error("Image data is None after loading attempts.")
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_err: logging.error(f"Error removing temporary file {temp_file_path}: {rm_err}")
            return render_template('register.html'), 500

        # Check for existing user *before* encoding face
        try:
            existing_user = mongo.db.students.find_one({'name': name})
            if existing_user:
                logging.warning(f"Registration attempt for existing user: {name}")
                flash(f"User '{name}' is already registered!", "warning")
                if temp_file_path and os.path.exists(temp_file_path):
                    try: os.remove(temp_file_path)
                    except OSError as rm_err: logging.error(f"Error removing temporary file {temp_file_path}: {rm_err}")
                return render_template('register.html'), 400 # Use 400 Bad Request for duplicate
        except Exception as e:
            logging.error(f"Database error checking for existing user {name}: {e}")
            flash("Database error checking for existing user.", "error")
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as rm_err: logging.error(f"Error removing temporary file {temp_file_path}: {rm_err}")
            return render_template('register.html'), 500


        logging.info(f"Processing registration for {name} from {image_source}.")
        face_embedding, box = detect_and_encode_face(image)

        # Clean up temp file immediately after use
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError as e: logging.error(f"Error removing upload file {temp_file_path} after processing: {e}")

        # Handle case where face detection/encoding fails
        if face_embedding is None:
            message = "No face detected in the provided image. Please ensure the face is clear and well-lit."
            if box: # If a box was detected but encoding failed
                message = "Face detected, but could not process it for registration. Try a clearer image or different angle."
            logging.warning(f"No face detected or embedding failed for registration image of {name}.")
            flash(message, "error")
            return render_template('register.html'), 400 # Use 400 for bad input image

        # Save user data to database
        try:
            embedding_list = face_embedding.tolist() # Convert numpy array to list for MongoDB
            user_data = {
                'name': name,
                'role': role,
                'face_embedding': embedding_list,
                'registered_at': datetime.utcnow(), # Store registration time in UTC
                'branch': branch,                  # Will be None if not applicable
                'academic_year': academic_year     # Will be None if not applicable
            }

            result = mongo.db.students.insert_one(user_data)
            logging.info(f"Successfully registered user: {name} (ID: {result.inserted_id}) with role: {role}, branch: {branch}, year: {academic_year}")
            flash(f"User '{name}' ({role.capitalize()}) registered successfully!", "success")

            load_known_faces() # Refresh the in-memory cache

            return redirect(url_for('view_students')) # Redirect after successful registration

        except Exception as e:
            logging.error(f"Database error during registration for {name}: {e}", exc_info=True)
            flash("Database error occurred during registration.", "error")
            return render_template('register.html'), 500 # Internal server error

    # Render the registration form for GET requests
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
        sort_by = request.args.get('sort_by', 'name') # Default sort by name
        sort_order = request.args.get('sort_order', 'asc') # Default sort order asc
        india_tz = pytz.timezone('Asia/Kolkata')

        # Fetch distinct branches efficiently
        distinct_branches = []
        try:
            # Query only for students and staff as 'others' don't have branches
            branch_query = {'branch': {'$nin': [None, '']}, 'role': {'$in': ['student', 'staff']}}
            distinct_branches = mongo.db.students.distinct('branch', branch_query)
            distinct_branches = sorted([b for b in distinct_branches if isinstance(b, str)])
        except Exception as db_err:
            logging.error(f"Error fetching distinct branches: {db_err}")
            flash("Error loading branch filter options.", "warning")

        # --- Build MongoDB Query ---
        query_conditions = []
        if search_query:
            # Case-insensitive regex search on name OR branch
            regex = {'$regex': search_query, '$options': 'i'}
            query_conditions.append({'$or': [{'name': regex}, {'branch': regex}]})
        if role_filter:
            query_conditions.append({'role': role_filter})
        # Apply branch filter only if role allows it (student/staff) or if no role is selected
        if branch_filter and (role_filter in ['student', 'staff'] or not role_filter):
             query_conditions.append({'branch': branch_filter})
        # Apply academic year filter only if role is student or no role is selected
        if academic_year_filter and (role_filter == 'student' or not role_filter):
             query_conditions.append({'academic_year': academic_year_filter})

        # Combine conditions with $and if multiple exist
        query = {'$and': query_conditions} if query_conditions else {}
        logging.debug(f"Executing student query: {query}")
        # --- End Query Build ---

        # --- Sorting ---
        mongo_sort_order = 1 if sort_order == 'asc' else -1
        # Validate sort field
        valid_sort_fields = ['name', 'role', 'registered_at', 'branch', 'academic_year']
        if sort_by not in valid_sort_fields: sort_by = 'name' # Default to name if invalid
        # --- End Sorting ---

        students_cursor = mongo.db.students.find(query).sort(sort_by, mongo_sort_order)
        students_list = list(students_cursor)
        logging.info(f"Found {len(students_list)} students matching filters.")

        # Process data for display
        for student in students_list:
            # Format registration date
            reg_at_utc = student.get('registered_at')
            if isinstance(reg_at_utc, datetime):
                 # Convert UTC to Kolkata Time for display
                 reg_at_kolkata = reg_at_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                 student['registered_at_str'] = reg_at_kolkata.strftime('%Y-%m-%d %H:%M:%S %Z')
            else: student['registered_at_str'] = "N/A" # Handle missing date

            # Convert ObjectId to string for URL generation
            student['_id_str'] = str(student['_id'])
            # Prepare display values, handling None or missing values
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
        # Attempt to load distinct branches even on error for filter dropdowns
        distinct_branches_on_error = []
        try:
            branch_query = {'branch': {'$nin': [None, '']}, 'role': {'$in': ['student', 'staff']}}
            distinct_branches_on_error = mongo.db.students.distinct('branch', branch_query)
            distinct_branches_on_error = sorted([b for b in distinct_branches_on_error if isinstance(b, str)])
        except: pass # Ignore errors fetching branches if main query failed

        # Pass back current filter values to repopulate form
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
         return redirect(url_for('view_students')) # Redirect back

    try:
        oid = ObjectId(student_id)
    except Exception:
        flash("Invalid student ID format.", "error")
        return redirect(url_for('view_students'))

    try:
        # Get student name before deleting for logging/messaging
        student_to_delete = mongo.db.students.find_one({'_id': oid}, {'name': 1})
        if not student_to_delete:
            flash("Student not found for deletion.", "error")
            return redirect(url_for('view_students')), 404 # Not Found status

        student_name = student_to_delete.get('name', f'ID_{student_id}') # Use name or ID for messages

        # Perform deletion
        result = mongo.db.students.delete_one({'_id': oid})

        if result.deleted_count > 0:
            logging.info(f"Successfully deleted student: {student_name} (ID: {student_id})")
            flash(f"Student '{student_name}' deleted successfully.", "success")
            # Refresh the known faces cache since a face was removed
            load_known_faces()

            # Attempt to remove related logs (optional, best-effort)
            try:
                # Remove only 'known_sighting' logs associated with this name
                log_result = mongo.db.seen_log.delete_many({'name': student_name, 'type': 'known_sighting'})
                if log_result.deleted_count > 0:
                    logging.info(f"Removed {log_result.deleted_count} sighting logs for deleted student {student_name}.")
            except Exception as log_e:
                logging.error(f"Error removing sighting logs for {student_name}: {log_e}")
                # Don't fail the whole operation, just warn the user
                flash("Student deleted, but failed to remove associated logs.", "warning")
        else:
            # This case should be rare if find_one succeeded, but handle defensively
            logging.warning(f"Deletion query for student {student_name} (ID: {student_id}) reported 0 deleted.")
            flash("Student found but could not be deleted.", "error")

    except Exception as e:
        logging.error(f"Error deleting student {student_id}: {e}", exc_info=True)
        flash("An error occurred while trying to delete the student.", "error")

    # Redirect back to the student list regardless of outcome
    return redirect(url_for('view_students'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, redirect to index
    if session.get('user_type') == 'admin':
        return redirect(url_for('index'))

    if request.method == 'POST':
        password = request.form.get('admin_password')
        # Load admin password from environment or use a default
        admin_pass = os.environ.get("ADMIN_PASSWORD", "0000") # Use a default if not set

        if password == admin_pass:
            session['user_type'] = 'admin'
            session['user_name'] = 'Admin' # Store generic admin name
            # Make session persistent for a duration
            session.permanent = True
            app.permanent_session_lifetime = timedelta(hours=8) # Example: 8-hour session
            logging.info("Admin login successful.")
            flash("Admin login successful.", "success")
            return redirect(url_for('index')) # Redirect to dashboard
        else:
            logging.warning("Invalid admin password attempt.")
            flash("Invalid admin credentials.", "error")
            return render_template('login.html'), 401 # Unauthorized status

    # Show login page for GET requests
    return render_template('login.html')


@app.route('/logout')
def logout():
    user_name = session.get('user_name', 'User') # Get username for logging
    session.clear() # Clear all session data
    logging.info(f"User '{user_name}' logged out.")
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('login')) # Redirect to login page


@app.before_request
def restrict_access():
    # Allow access to login, logout, and static files without logging in
    if request.endpoint in ['login', 'static', 'logout']:
        return None # Skip check for these endpoints

    # If user is not logged in (no user_type in session)
    if 'user_type' not in session:
        flash("Please log in to access this page.", "info")
        return redirect(url_for('login'))

    # If user is logged in as admin, allow access
    if session.get('user_type') == 'admin':
        return None # Allow access

    # If logged in but not as admin (future proofing for other roles?)
    # For now, only admin access is implemented, so this case implies unauthorized access
    flash("Admin access required. Please log in.", "warning")
    return redirect(url_for('login'))


@app.route('/seen_log')
def seen_log_view():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))
    try:
        # Fetch all log records, sorted by timestamp descending
        log_records_cursor = mongo.db.seen_log.find().sort('timestamp', -1)
        log_records = list(log_records_cursor)
        logging.debug(f"Fetched {len(log_records)} seen log records.")

        grouped_logs = defaultdict(list)
        india_tz = pytz.timezone('Asia/Kolkata') # Use consistent timezone

        # Process and group records
        for record in log_records:
            # Format timestamp
            ts_utc = record.get('timestamp')
            if isinstance(ts_utc, datetime):
                ts_kolkata = ts_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                record['timestamp_str'] = ts_kolkata.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                record['timestamp_str'] = "Invalid Date"

            # Determine group key based on type
            record_type = record.get('type')
            group_key = "Other" # Default group
            if record_type == 'known_sighting':
                group_key = record.get('name', 'Known (Error)')
                if not record.get('name'): record['name'] = 'Known (Error)' # Handle missing name
            elif record_type in ['unknown_sighting', 'processing_error', 'processing_error_sighting']: # Include potential error types
                group_key = 'Unknown'
                # Ensure image field exists, even if None
                if 'face_image_base64' not in record: record['face_image_base64'] = None
            # Add other types if necessary

            grouped_logs[group_key].append(record)

        # Sort the groups: Known (by name), then Unknown, then Others
        def sort_key(item):
            key = item[0] # The group key (name, 'Unknown', 'Other')
            if key == 'Unknown': return (1, key) # Unknown comes after known
            if key == 'Other': return (2, key)   # Other comes last
            if key == 'Known (Error)': return (0, '~') # Error case first among known
            return (0, key.lower()) # Sort known names case-insensitively

        sorted_grouped_logs = dict(sorted(grouped_logs.items(), key=sort_key))

        return render_template('seen_log.html', grouped_logs=sorted_grouped_logs)

    except Exception as e:
        logging.error(f"Error fetching or grouping seen log records: {e}", exc_info=True)
        flash("Error loading seen log data.", "error")
        return render_template('seen_log.html', grouped_logs={}), 500 # Return empty on error


@app.route('/unknown_captures')
def unknown_captures():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        # Get time filters from query parameters
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')
        time_filter = {}
        india_tz = pytz.timezone('Asia/Kolkata')
        start_time_value = '' # Store original value for form repopulation
        end_time_value = ''   # Store original value for form repopulation

        # Build timestamp query if filters are provided
        if start_time_str:
            try:
                # Parse HTML datetime-local format (YYYY-MM-DDTHH:MM)
                start_dt_local = datetime.fromisoformat(start_time_str)
                # Make it timezone-aware (assuming input is Kolkata time)
                start_dt_aware_local = india_tz.localize(start_dt_local)
                # Convert to UTC for MongoDB query
                time_filter['$gte'] = start_dt_aware_local.astimezone(pytz.utc)
                start_time_value = start_time_str # Keep original string for form
            except (ValueError, TypeError) as e:
                flash(f"Invalid start time format ignored: {e}", "warning")
        if end_time_str:
            try:
                end_dt_local = datetime.fromisoformat(end_time_str)
                end_dt_aware_local = india_tz.localize(end_dt_local)
                # Convert to UTC for MongoDB query
                time_filter['$lte'] = end_dt_aware_local.astimezone(pytz.utc)
                end_time_value = end_time_str # Keep original string for form
            except (ValueError, TypeError) as e:
                flash(f"Invalid end time format ignored: {e}", "warning")

        # Base query for unknown/error types
        query = {"type": {"$in": ["unknown_sighting", "processing_error", "processing_error_sighting"]}}
        # Add time filter if present
        if time_filter:
            query['timestamp'] = time_filter
        logging.debug(f"Unknown captures query: {query}")

        # Execute query, sort by timestamp descending
        log_records_cursor = mongo.db.seen_log.find(query).sort('timestamp', -1)
        log_records = list(log_records_cursor)
        logging.debug(f"Fetched {len(log_records)} unknown capture records.")

        # Group logs by date (Kolkata time)
        grouped_unknowns = defaultdict(list)
        for record in log_records:
            ts_utc = record.get('timestamp')
            if isinstance(ts_utc, datetime):
                ts_kolkata = ts_utc.replace(tzinfo=pytz.utc).astimezone(india_tz)
                record['timestamp_str'] = ts_kolkata.strftime('%Y-%m-%d %H:%M:%S %Z')
                date_str = ts_kolkata.strftime('%Y-%m-%d') # Group by date
            else:
                record['timestamp_str'] = "Invalid Date"
                date_str = "Unknown Date"
            # Ensure image field exists
            if 'face_image_base64' not in record: record['face_image_base64'] = None
            grouped_unknowns[date_str].append(record)

        # Convert defaultdict to regular dict for template
        grouped_unknowns_dict = dict(grouped_unknowns)

        # Pass filter values back to template for form repopulation
        # Ensure they are in the correct 'datetime-local' format if needed (already are)
        return render_template('unknown_captures.html',
                                grouped_unknowns=grouped_unknowns_dict,
                                start_time_value=start_time_value, # Pass original strings
                                end_time_value=end_time_value)

    except Exception as e:
        logging.error(f"Error fetching or grouping unknown captures: {e}", exc_info=True)
        flash("Error loading unknown captures data.", "error")
        return render_template('unknown_captures.html', grouped_unknowns={}, start_time_value='', end_time_value=''), 500


@app.route('/reset_seen_log', methods=['POST'])
def reset_seen_log():
    if session.get('user_type') != 'admin':
        logging.warning("Unauthorized attempt to reset seen log.")
        flash("Unauthorized action.", "error")
        return redirect(url_for('login')) # Redirect if not admin

    try:
        # Delete all documents from the seen_log collection
        result = mongo.db.seen_log.delete_many({})
        deleted_count = result.deleted_count
        logging.info(f"Seen log reset requested by admin. {deleted_count} records deleted.")
        flash(f"Seen log reset successfully. {deleted_count} records were deleted.", "success")
    except Exception as e:
        logging.error(f"Error resetting seen log: {e}", exc_info=True)
        flash("An error occurred while resetting the seen log.", "error")

    # Redirect back to the seen log view page
    return redirect(url_for('seen_log_view'))


@app.route('/attendance', methods=['GET', 'POST'])
def attendance_page():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    india_tz = pytz.timezone('Asia/Kolkata')
    now_india = datetime.now(india_tz)
    today_date_str = now_india.strftime('%Y-%m-%d')

    # Get filters from request args (for GET) or form (for POST)
    # Use request.values to handle both GET and POST easily for filters
    branch_filter = request.values.get('branch_filter', '')
    academic_year_filter = request.values.get('academic_year_filter', '')
    # Get selected date, default to today
    selected_date = request.values.get('attendance_date', today_date_str)

    attendance_records = []

    try:
        # Validate and parse the selected date
        try:
            selected_date_obj = datetime.strptime(selected_date, "%Y-%m-%d").date()
        except ValueError:
            logging.warning(f"Invalid date format '{selected_date}'. Defaulting to today.")
            selected_date_obj = now_india.date()
            selected_date = today_date_str # Update selected_date string if defaulted

        # Calculate UTC start and end of the selected day in Kolkata time
        start_of_day_local = india_tz.localize(datetime.combine(selected_date_obj, time.min))
        end_of_day_local = india_tz.localize(datetime.combine(selected_date_obj, time.max))
        start_of_day_utc = start_of_day_local.astimezone(pytz.utc)
        end_of_day_utc = end_of_day_local.astimezone(pytz.utc)

        logging.info(f"Attendance Query: Date={selected_date}, Branch='{branch_filter}', Year='{academic_year_filter}'")
        logging.info(f"Attendance Query: Fetching logs between {start_of_day_utc} (UTC) and {end_of_day_utc} (UTC)")

        # --- Dynamic Collection Logic (if branch and year are selected) ---
        # <<< NOTE: This line ensures we query the correct collection >>>
        collection_to_query = mongo.db.seen_log # Default to main log
        log_query = {
            'type': 'known_sighting',
            'timestamp': {'$gte': start_of_day_utc, '$lte': end_of_day_utc}
        }

        # Prefetch user details (name -> branch, year) for filtering
        user_details = {}
        try:
            known_users = mongo.db.students.find({}, {"name": 1, "branch": 1, "academic_year": 1, "_id": 0})
            for user in known_users:
                user_details[user['name']] = {
                    'branch': user.get('branch'), # Keep None if missing
                    'academic_year': user.get('academic_year') # Keep None if missing
                }
        except Exception as db_err:
            logging.error(f"Error prefetching user details: {db_err}")
            # Continue without details, filtering might be less accurate


        # Fetch relevant log records
        log_records_cursor = collection_to_query.find(log_query).sort('timestamp', 1) # Sort ascending for first/last

        # Process logs to find first (arriving) and last (leaving) times per person
        attendance_data = defaultdict(lambda: {'first': None, 'last': None})
        processed_log_count = 0
        for record in log_records_cursor:
            name = record.get('name')
            timestamp_utc = record.get('timestamp')
            processed_log_count +=1

            if name and isinstance(timestamp_utc, datetime):
                details = user_details.get(name)
                # Apply filters *before* storing timestamps
                if details:
                    branch_matches = (not branch_filter) or (details['branch'] == branch_filter)
                    year_matches = (not academic_year_filter) or (details['academic_year'] == academic_year_filter)

                    if branch_matches and year_matches:
                        if attendance_data[name]['first'] is None:
                            attendance_data[name]['first'] = timestamp_utc
                        attendance_data[name]['last'] = timestamp_utc # Always update last seen
                    # else: # Filtered out
                        # logging.debug(f"Filtering out log for '{name}' due to mismatch: Branch OK={branch_matches}, Year OK={year_matches}")
                # else: # User details not found, cannot filter effectively
                #     logging.warning(f"Details not found for user '{name}' in log record. Including by default if filters are off.")
                #     Include if no filters are active
                elif not branch_filter and not academic_year_filter: # Check if details missing AND filters off
                    if attendance_data[name]['first'] is None:
                        attendance_data[name]['first'] = timestamp_utc
                    attendance_data[name]['last'] = timestamp_utc


        logging.info(f"Attendance Query: Processed {processed_log_count} known_sighting logs for {selected_date}.")
        logging.info(f"Attendance Processing: Found first/last times for {len(attendance_data)} individuals matching filters.")

        # Format the final attendance records
        final_attendance = []
        for name, times in attendance_data.items():
            if times['first'] and times['last']: # Ensure both times were recorded
                arriving_time_kolkata = times['first'].replace(tzinfo=pytz.utc).astimezone(india_tz)
                leaving_time_kolkata = times['last'].replace(tzinfo=pytz.utc).astimezone(india_tz)
                # Get details again, handling potential missing users safely
                details = user_details.get(name, {'branch': 'N/A', 'academic_year': 'N/A'})

                final_attendance.append({
                    'name': name,
                    'branch': details['branch'] or 'N/A', # Handle None
                    'academic_year': details['academic_year'] or 'N/A', # Handle None
                    'arriving_time': arriving_time_kolkata.strftime('%H:%M:%S'),
                    'leaving_time': leaving_time_kolkata.strftime('%H:%M:%S'),
                    'date': arriving_time_kolkata.strftime('%Y-%m-%d') # Date based on arriving time
                })

        # Sort final list by name
        final_attendance.sort(key=lambda x: x['name'])
        attendance_records = final_attendance
        logging.info(f"Generated final attendance list with {len(attendance_records)} records for {selected_date}.")

    except Exception as e:
        logging.error(f"Error generating attendance page: {e}", exc_info=True)
        flash("Error loading attendance data.", "error")
        attendance_records = [] # Ensure it's empty on error

    # Render the template with data and filter values
    return render_template('attendance.html',
                           attendance_records=attendance_records,
                           branch_filter=branch_filter, # Pass current filters back
                           academic_year_filter=academic_year_filter,
                           today_date=today_date_str,
                           selected_date=selected_date) # Pass selected date back


@app.route('/export_attendance_excel')
def export_attendance_excel():
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        # Get filters from query parameters (passed from the attendance page link)
        branch_filter = request.args.get('branch_filter', '')
        academic_year_filter = request.args.get('academic_year_filter', '')
        # Get the date for which to export, default to today
        selected_date_str = request.args.get('date', datetime.now(india_tz).strftime('%Y-%m-%d'))

        # Validate and parse the selected date
        try:
            selected_date_obj = datetime.strptime(selected_date_str, "%Y-%m-%d").date()
        except ValueError:
            logging.warning(f"Invalid date format '{selected_date_str}' for export. Defaulting to today.")
            selected_date_obj = datetime.now(india_tz).date()
            selected_date_str = selected_date_obj.strftime('%Y-%m-%d')

        # --- Re-calculate attendance data for the selected date and filters ---
        start_of_day_local = india_tz.localize(datetime.combine(selected_date_obj, time.min))
        end_of_day_local = india_tz.localize(datetime.combine(selected_date_obj, time.max))
        start_of_day_utc = start_of_day_local.astimezone(pytz.utc)
        end_of_day_utc = end_of_day_local.astimezone(pytz.utc)

        # <<< NOTE: This line ensures we query the correct collection >>>
        collection_to_query = mongo.db.seen_log
        log_query = {
            'type': 'known_sighting',
            'timestamp': {'$gte': start_of_day_utc, '$lte': end_of_day_utc}
        }

        user_details = {}
        known_users = mongo.db.students.find({}, {"name": 1, "branch": 1, "academic_year": 1, "_id": 0})
        for user in known_users:
            user_details[user['name']] = {'branch': user.get('branch'), 'academic_year': user.get('academic_year')}

        log_records_cursor = collection_to_query.find(log_query).sort('timestamp', 1)

        attendance_data = defaultdict(lambda: {'first': None, 'last': None})
        for record in log_records_cursor:
            name = record.get('name')
            timestamp_utc = record.get('timestamp')
            if name and isinstance(timestamp_utc, datetime):
                 details = user_details.get(name)
                 if details:
                     branch_matches = (not branch_filter) or (details['branch'] == branch_filter)
                     year_matches = (not academic_year_filter) or (details['academic_year'] == academic_year_filter)
                     if branch_matches and year_matches:
                         if attendance_data[name]['first'] is None:
                             attendance_data[name]['first'] = timestamp_utc
                         attendance_data[name]['last'] = timestamp_utc
                 elif not branch_filter and not academic_year_filter: # Include if no details and no filters
                      if attendance_data[name]['first'] is None:
                         attendance_data[name]['first'] = timestamp_utc
                      attendance_data[name]['last'] = timestamp_utc


        final_attendance = []
        for name, times in attendance_data.items():
            if times['first'] and times['last']:
                arriving_time_kolkata = times['first'].replace(tzinfo=pytz.utc).astimezone(india_tz)
                leaving_time_kolkata = times['last'].replace(tzinfo=pytz.utc).astimezone(india_tz)
                details = user_details.get(name, {'branch': 'N/A', 'academic_year': 'N/A'})
                final_attendance.append({
                    'name': name, 'branch': details['branch'] or 'N/A', 'academic_year': details['academic_year'] or 'N/A',
                    'arriving_time': arriving_time_kolkata.strftime('%H:%M:%S'),
                    'leaving_time': leaving_time_kolkata.strftime('%H:%M:%S'),
                    'date': arriving_time_kolkata.strftime('%Y-%m-%d')
                })
        final_attendance.sort(key=lambda x: x['name'])
        # --- End Recalculation ---

        # --- Generate Excel File ---
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        # Create a clean title based on date and filters
        sheet_title_base = f"Attendance_{selected_date_str}"
        if branch_filter: sheet_title_base += f"_{branch_filter.replace('.', '').replace(' ', '_')}"
        if academic_year_filter: sheet_title_base += f"_{academic_year_filter.replace('.', '').replace(' ', '_')}"
        # Ensure title length is within Excel limits (31 chars)
        sheet.title = sheet_title_base[:31]

        headers = ["Name", "Branch", "Academic Year", "Arriving Time", "Leaving Time", "Date"]
        sheet.append(headers)
        for record in final_attendance:
            sheet.append([record['name'], record['branch'], record['academic_year'],
                          record['arriving_time'], record['leaving_time'], record['date']])

        # Auto-adjust column widths
        for col in sheet.columns:
             max_length = 0
             column_letter = col[0].column_letter # Get column letter
             for cell in col:
                 try:
                     if cell.value: # Check if cell has value
                         # Calculate length of string representation
                         current_length = len(str(cell.value))
                         if current_length > max_length:
                             max_length = current_length
                 except: pass # Ignore errors for problematic cells
             # Set adjusted width (max length + buffer)
             adjusted_width = (max_length + 2)
             sheet.column_dimensions[column_letter].width = adjusted_width

        # Save workbook to a BytesIO stream
        excel_stream = BytesIO()
        workbook.save(excel_stream)
        excel_stream.seek(0) # Rewind stream to the beginning
        logging.info(f"Generated Excel export for {len(final_attendance)} records for date {selected_date_str}.")

        # Create filename based on date and filters
        filename_base = f"attendance_{selected_date_str}"
        if branch_filter: filename_base += f"_{branch_filter.replace('.', '').replace(' ', '_')}"
        if academic_year_filter: filename_base += f"_{academic_year_filter.replace('.', '').replace(' ', '_')}"
        filename = f"{filename_base}.xlsx"

        # Return the stream as a downloadable file
        return Response(
            excel_stream,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment;filename="{filename}"'}
        )

    except Exception as e:
        logging.error(f"Error generating Excel export: {e}", exc_info=True)
        flash("Error generating Excel export.", "error")
        # Redirect back to attendance page, preserving filters
        return redirect(url_for('attendance_page',
                                branch_filter=request.args.get('branch_filter',''),
                                academic_year_filter=request.args.get('academic_year_filter',''),
                                attendance_date=request.args.get('date', '')))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Disable Flask's default reloader in production/threaded mode if debug is False
    use_reloader = app.debug
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=use_reloader)