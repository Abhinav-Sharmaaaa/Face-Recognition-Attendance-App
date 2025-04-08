from flask import Flask, render_template, request, redirect, url_for, jsonify, session, Response, flash
import os
from datetime import datetime, timedelta, time # Added time
import pytz # Kept for seen_log timestamps
from collections import defaultdict # Import defaultdict for easier grouping
import cv2
import numpy as np
import io
from io import BytesIO # Added for Excel export
import logging
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import base64
import urllib.parse
from scipy.spatial.distance import cosine
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId # Added for MongoDB ID handling
import json # Added for configuration
import openpyxl # Added for Excel export

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

# --- Model Loading (Error handling remains) ---
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
# --- Configuration Handling ---
CONFIG_FILE = 'config.json'

def load_config():
    """Loads configuration from JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {CONFIG_FILE}. Using defaults.")
            return {} # Return empty dict if file is corrupt
        except Exception as e:
            logging.error(f"Error reading config file {CONFIG_FILE}: {e}")
            return {}
    return {} # Return empty dict if file doesn't exist

def save_config(config_data):
    """Saves configuration to JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        logging.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        logging.error(f"Error writing config file {CONFIG_FILE}: {e}")
        flash("Error saving configuration.", "error") # Flash error to user

try:
    face_recognizer = cv2.FaceRecognizerSF.create(
        model=sface_model_path,
        config="",
        backend_id=0, # 0 for default, 3 for CUDA if available and OpenCV built with it
        target_id=0 # 0 for CPU, 1 for CUDA
    )
    logging.info("SFace face recognizer loaded successfully.")
except cv2.error as e:
    logging.error(f"Error loading SFace model: {e}")
    exit("Failed to load face recognizer.")
except AttributeError:
     logging.error("cv2.FaceRecognizerSF not found. Check OpenCV installation (needs version >= 4.5.4).")
     exit("OpenCV FaceRecognizerSF unavailable.")


# --- Known Face Data Handling ---
known_face_data = {} # Cache: name -> {'embedding': np.array, 'branch': str, 'role': str} # Added role
known_face_embeddings_list = [] # List of embeddings for faster comparison
known_face_names = [] # List of names corresponding to embeddings list

RECOGNITION_THRESHOLD = 0.36 # SFace threshold (lower is stricter)

def load_known_faces():
    global known_face_data, known_face_embeddings_list, known_face_names
    logging.info("Loading known faces from database...")
    known_face_data = {}
    temp_embeddings_list = []
    temp_names_list = []
    try:
        # Fetch necessary fields including role
        # Using 'students' collection name (consider renaming to 'users' later)
        users_cursor = mongo.db.students.find({}, {"name": 1, "branch": 1, "role": 1, "face_embedding": 1}) # Added role
        count = 0
        for user_doc in users_cursor:
            # Check if essential data exists (name and embedding are crucial)
            if 'face_embedding' in user_doc and user_doc['face_embedding'] and 'name' in user_doc:
                try:
                    # Convert embedding from list (stored in MongoDB) to NumPy array
                    embedding_np = np.array(user_doc['face_embedding']).astype(np.float32)
                    # Validate shape (SFace generates 128-dim features)
                    if embedding_np.shape == (128,):
                        known_face_data[user_doc['name']] = {
                            'embedding': embedding_np,
                            'branch': user_doc.get('branch'), # Get branch (might be None)
                            'role': user_doc.get('role', 'Unknown') # Get role, default if missing
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

        known_face_embeddings_list = temp_embeddings_list # Update global list
        known_face_names = temp_names_list # Update global list
        logging.info(f"Loaded {count} known faces (embeddings) into memory cache.")

    except Exception as e:
        logging.error(f"Error loading known faces from database: {e}")

# --- Face Detection and Encoding (Unchanged) ---
def detect_and_encode_face(image_np):
    """
    Detects the most prominent face using YuNet and generates its SFace embedding.
    """
    if image_np is None or image_np.size == 0:
        logging.warning("detect_and_encode_face received an empty image.")
        return None, None
    if face_detector is None or face_recognizer is None:
        logging.error("Detector or Recognizer not initialized.")
        return None, None

    height, width, _ = image_np.shape
    # Set the input size for the detector
    face_detector.setInputSize((width, height))

    try:
        # Detect faces
        status, faces = face_detector.detect(image_np)
    except cv2.error as e:
        logging.error(f"Error during face detection: {e}")
        return None, None

    if faces is None or len(faces) == 0:
        # logging.debug("No faces detected in the image.")
        return None, None

    # Assume the first detected face is the target (or implement logic to choose)
    # YuNet returns faces as [x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
    # Find face with the highest score (last element)
    best_face_index = np.argmax(faces[:, -1])
    face_data = faces[best_face_index]
    box = face_data[0:4].astype(np.int32)
    (x, y, w, h) = box

    # Ensure bounding box is within image boundaries before cropping/alignment
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    if w <= 0 or h <= 0:
        logging.warning(f"Invalid bounding box after boundary check: x={x}, y={y}, w={w}, h={h}")
        return None, (x,y,w,h) # Return invalid box for potential drawing

    try:
        # Align and crop the detected face using landmarks provided by YuNet
        # face_data includes the necessary landmark points for alignCrop
        aligned_face = face_recognizer.alignCrop(image_np, face_data)

        if aligned_face is None or aligned_face.size == 0:
            logging.warning("Face alignment/cropping failed.")
            return None, (x,y,w,h) # Return box even if alignment fails

        # Generate the feature embedding for the aligned face
        face_embedding = face_recognizer.feature(aligned_face)
        # Feature returns a (1, 128) array, flatten to (128,)
        face_embedding_flat = face_embedding.flatten()

        # logging.debug(f"Generated embedding of shape: {face_embedding_flat.shape}")
        return face_embedding_flat, (x, y, w, h)

    except cv2.error as e:
        logging.error(f"Error during face alignment or feature extraction: {e}")
        return None, (x,y,w,h) # Return box even if error occurs
    except Exception as e:
        logging.error(f"Unexpected error during encoding: {e}")
        return None, (x,y,w,h)


load_known_faces() # Load faces on startup

# --- Routes ---

@app.route('/refresh_faces', methods=['POST'])
def refresh_faces_route():
    """ Admin-only endpoint to manually refresh the known face cache """
    if session.get('user_type') != 'admin':
        return jsonify({"message": "Unauthorized"}), 403
    try:
        load_known_faces() # Reload from DB
        flash(f"Face cache refreshed. Loaded {len(known_face_data)} faces.", "success")
        return jsonify({"message": f"Face cache refreshed. Loaded {len(known_face_data)} faces."}), 200
    except Exception as e:
        logging.error(f"Error during manual face cache refresh: {e}")
        flash("Error refreshing face cache.", "error")
        return jsonify({"message": "Error refreshing face cache."}), 500

@app.route('/')
def index():
    """ Main dashboard for admin """
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))
    # Admin dashboard
    return render_template('index.html') # Assumes index.html is the admin dashboard

@app.route('/live_feed')
def live_feed():
    """ Page to display the live camera feed """
    # Access control: Only admin can view the live feed page
    if session.get('user_type') != 'admin':
        flash("Unauthorized access to live feed.", "warning")
        return redirect(url_for('login'))
    return render_template('live_feed.html')


def generate_frames():
    """ Generates frames from webcam, performs face recognition, logs sightings, and stops at configured time if enabled. """
    config = load_config() # Load configuration
    stop_time_enabled = config.get('stop_time_enabled', False) # Default to disabled if not set
    stop_time_str = config.get('stop_time')
    stop_time_obj = None

    if stop_time_enabled and stop_time_str: # Only process stop time if enabled and set
        try:
            stop_time_obj = datetime.strptime(stop_time_str, '%H:%M').time()
            logging.info(f"Live feed monitoring enabled to stop at {stop_time_str}")
        except ValueError:
            logging.warning(f"Invalid stop_time format '{stop_time_str}' in config. Ignoring stop time despite being enabled.")
            stop_time_obj = None # Ensure it's None if format is bad
    elif stop_time_enabled:
        logging.warning("Stop time is enabled in config, but no valid time is set.")
    else:
        logging.info("Automatic live feed stop time is disabled.")


    cap = cv2.VideoCapture(0) # Use camera 0
    if not cap.isOpened():
        logging.error("Cannot open camera 0")
        return

    frame_count = 0
    process_every_n_frames = 3 # Process every 3rd frame to save resources
    last_known_faces = {} # Store last detected faces for smoother display
    india_tz = pytz.timezone('Asia/Kolkata') # Timezone for logging
    seen_log_collection = mongo.db.seen_log # Collection for logging ALL sightings
    last_log_times = {} # Dictionary to track last log time for each KNOWN person
    log_interval = timedelta(minutes=1) # Set the minimum interval between logs for KNOWN faces
    last_unknown_log_time = None # Timestamp of the last logged unknown face
    unknown_log_interval_seconds = config.get('unknown_log_interval_seconds', 60) # Default to 60 seconds if not set
    unknown_log_interval = timedelta(seconds=unknown_log_interval_seconds)
    logging.info(f"Unknown face log interval set to: {unknown_log_interval_seconds} seconds.")


    while True:
        success, frame = cap.read()
        if not success:
            logging.warning("Failed to read frame from camera.")
            break

        # --- Stop Time Check (only if enabled and time object is valid) ---
        if stop_time_enabled and stop_time_obj:
            india_tz = pytz.timezone('Asia/Kolkata') # Ensure timezone is defined before use
            current_time_india = datetime.now(india_tz).time()
            if current_time_india >= stop_time_obj:
                logging.info(f"Current time {current_time_india} reached or passed enabled stop time {stop_time_obj}. Stopping live feed.")
                break # Exit the loop
        # --- End Stop Time Check ---

        # Create a copy for processing, display the flipped original
        processing_frame = frame.copy()
        frame = cv2.flip(frame, 1) # Flip horizontally for mirror effect
        processing_frame = cv2.flip(processing_frame, 1) # Process the flipped frame too
        frame_count += 1

        current_detected_faces = {} # Faces detected in THIS frame

        # Process frame only periodically
        if frame_count % process_every_n_frames == 0:
            face_embedding, box = detect_and_encode_face(processing_frame)
            status_text = ""
            status_color = (0, 0, 0) # Black default

            if box: # A face bounding box was returned (even if encoding failed)
                current_time_india = datetime.now(india_tz)
                recognized_user_name = None # Track if a known user was recognized

                if face_embedding is not None:
                    # Compare detected embedding with known embeddings
                    best_match_name = None
                    best_similarity = float('inf') # Lower cosine distance is better

                    # Iterate through cached embeddings and names
                    for i, stored_embedding in enumerate(known_face_embeddings_list):
                        try:
                            # Ensure shapes match before comparison
                            if face_embedding.shape != stored_embedding.shape:
                                logging.warning(f"Shape mismatch: current {face_embedding.shape}, stored {stored_embedding.shape}")
                                continue

                            similarity = cosine(face_embedding, stored_embedding)

                            if similarity < best_similarity:
                                best_similarity = similarity
                                best_match_name = known_face_names[i]

                        except Exception as e:
                            logging.error(f"Error comparing embedding {i} for {known_face_names[i]}: {e}")
                            continue # Skip to next known face on error

                    # Check if the best match meets the recognition threshold
                    if best_match_name and best_similarity < RECOGNITION_THRESHOLD:
                        recognized_user_name = best_match_name
                        logging.info(f"Live Feed Match: {recognized_user_name} (Cosine Dist: {best_similarity:.4f})")
                        status_text = f"Recognized: {recognized_user_name}"
                        status_color = (0, 255, 0) # Green for recognized

                    elif best_match_name: # Matched someone, but below threshold
                        status_text = f"Low Match: {best_match_name} ({best_similarity:.2f})"
                        status_color = (0, 165, 255) # Orange for low match
                    else: # No match found or no known faces loaded
                        status_text = "Unknown Face"
                        status_color = (255, 0, 0) # Red for unknown

                    current_detected_faces[0] = {'box': box, 'status': status_text, 'color': status_color}

                else: # box exists, but embedding failed
                     status_text = "Processing Error"
                     status_color = (255, 0, 255) # Magenta for error
                     current_detected_faces[0] = {'box': box, 'status': status_text, 'color': status_color}

                # --- Logging Sighting Logic ---
                if box: # Only log if a face box was detected
                    log_entry = {
                        'timestamp': current_time_india,
                        'status_at_log': status_text # Log the status determined above
                    }

                    # --- Known Face Logging ---
                    if recognized_user_name:
                        now = current_time_india
                        last_logged = last_log_times.get(recognized_user_name)
                        if last_logged is None or (now - last_logged) >= log_interval:
                            log_entry['type'] = 'known_sighting'
                            log_entry['name'] = recognized_user_name
                            try:
                                seen_log_collection.insert_one(log_entry)
                                last_log_times[recognized_user_name] = now
                                logging.info(f"Logged sighting of known face: {recognized_user_name} (Interval passed or first time)")
                            except Exception as db_err:
                                logging.error(f"Error inserting known sighting log for {recognized_user_name}: {db_err}")

                    # --- Unknown/Low Match/Error Face Logging (with Interval) ---
                    # Log if status is "Unknown Face", "Low Match", or "Processing Error"
                    elif status_text == "Unknown Face" or status_text.startswith("Low Match") or status_text == "Processing Error":
                        now = current_time_india
                        # Check if interval has passed since last unknown log
                        if last_unknown_log_time is None or (now - last_unknown_log_time) >= unknown_log_interval:
                            # Determine log type based on status
                            if status_text == "Processing Error":
                                log_entry['type'] = 'processing_error_sighting' # More specific type
                                log_entry['error_details'] = "Failed to generate face embedding" # Add context
                            else:
                                log_entry['type'] = 'unknown_sighting'

                            # Optionally add cropped face image (attempt even for errors if box exists)
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
                                 # Clear potentially partial image data if error occurred
                                 log_entry.pop('face_image_base64', None)

                            # Attempt to insert the log entry AFTER image processing attempt
                            try:
                                seen_log_collection.insert_one(log_entry)
                                last_unknown_log_time = now # Update last log time
                                logging.info(f"Logged unknown/low match face sighting ({status_text}) {'with image' if 'face_image_base64' in log_entry else 'without image'}. Interval passed.")
                            except Exception as db_err:
                                logging.error(f"Error inserting unknown/low match sighting log into DB: {db_err}")
                        # else: # Optional: Log that an unknown face was seen but not logged due to interval
                        #     logging.debug(f"Unknown/low match face detected ({status_text}), but not logged due to interval.")

                    # --- Other Statuses (Currently not logged) ---

                # --- End Logging Sighting Logic ---

            # Update last known faces for display smoothing
            last_known_faces = current_detected_faces if box else {} # Use current if box detected, else clear

            display_faces = current_detected_faces

        else:
            # On frames that are not processed, display the last known faces
            display_faces = last_known_faces

        # Draw boxes and status text on the frame to be displayed
        for face_id, face_info in display_faces.items():
             (x, y, w, h) = face_info['box']
             cv2.rectangle(frame, (x, y), (x + w, y + h), face_info['color'], 2)
             # Put text above the box
             cv2.putText(frame, face_info['status'], (x, y - 10 if y > 10 else y + 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_info['color'], 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.warning("Failed to encode frame to JPEG.")
            continue
        frame_bytes = buffer.tobytes()

        # Yield the frame in the multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    logging.info("Camera released.")


@app.route('/video_feed')
def video_feed():
    """ Endpoint to stream the processed video frames. """
    # Access control: Only admin can access the raw video feed stream
    if session.get('user_type') != 'admin':
        logging.warning("Unauthorized attempt to access video_feed endpoint.")
        # Return a response indicating forbidden access
        return Response("Unauthorized", status=403)

    # Return the generator function wrapped in a Response object
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """ Handles new user registration (Admin only) """
    if session.get('user_type') != 'admin':
        flash("Only admins can register new users.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form.get('name')
        role = request.form.get('role') # Added role field
        # Convert role to lowercase for consistent validation and storage
        role = role.lower() if role else None
        branch = request.form.get('branch') # Or department, group etc.
        photo_data = request.form.get('photo') # From webcam capture
        uploaded_file = request.files.get('uploadPhoto') # From file upload

        # Validate role
        allowed_roles = ['student', 'staff', 'others']
        if not role or role not in allowed_roles:
            flash(f"Invalid or missing role. Must be one of: {', '.join(allowed_roles)}.", "error")
            return render_template('register.html'), 400

        # Validate required fields based on role
        if not name:
             flash("Name is required.", "error")
             return render_template('register.html'), 400
        if role in ['student', 'staff'] and not branch:
            flash("Branch/Department is required for students and staff.", "error")
            return render_template('register.html'), 400
        # For 'others', branch is optional/ignored

        image = None
        image_source = "unknown"
        file_path = None # To keep track of temporary uploaded file

        # --- Image Loading Logic (Webcam or Upload) ---
        if photo_data and photo_data.startswith('data:image/'): # Check if it's base64 image data
            try:
                # Decode base64 image
                header, encoded = photo_data.split(',', 1)
                image_bytes = io.BytesIO(base64.b64decode(encoded))
                nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image_source = "webcam"
                logging.info("Processing image from webcam data.")
            except Exception as e:
                logging.error(f"Error decoding base64 image: {e}")
                flash("Invalid photo data from webcam.", "error")
                return render_template('register.html'), 400
        elif uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                uploaded_file.save(file_path)
                image = cv2.imread(file_path) # Read the saved image using OpenCV
                if image is None: # Check if imread failed
                    raise ValueError(f"cv2.imread returned None for {filename}. Check file format/integrity.")
                image_source = f"upload ({filename})"
                logging.info(f"Processing image from uploaded file: {filename}")
            except Exception as e:
                logging.error(f"Error saving or reading uploaded file '{filename}': {e}")
                flash(f"Could not process uploaded file: {e}", "error")
                 # Clean up temporary file if it exists
                if file_path and os.path.exists(file_path):
                     try: os.remove(file_path)
                     except OSError as rm_err: logging.error(f"Error removing temporary file {file_path}: {rm_err}")
                return render_template('register.html'), 400
        else:
            flash("No photo provided (either webcam capture or file upload is required).", "error")
            return render_template('register.html'), 400

        # --- Final Image Check ---
        if image is None:
            logging.error("Image data is None after loading attempts.")
            # Clean up if upload was attempted and failed after saving
            if file_path and os.path.exists(file_path):
                try: os.remove(file_path)
                except OSError as rm_err: logging.error(f"Error removing file {file_path} after load fail: {rm_err}")
            flash("Failed to load image data.", "error")
            return render_template('register.html'), 500

        # --- Check if user already exists ---
        # Using 'students' collection name as per original code
        existing_user = mongo.db.students.find_one({'name': name})
        if existing_user:
            logging.warning(f"Registration attempt for existing user: {name}")
            flash(f"User '{name}' is already registered!", "warning")
             # Clean up uploaded file if it exists
            if file_path and os.path.exists(file_path):
                try: os.remove(file_path)
                except OSError as e: logging.error(f"Error removing upload file {file_path} for existing user: {e}")
            return render_template('register.html'), 400 # Use 400 for client error (duplicate)

        # --- Process Face Embedding ---
        logging.info(f"Processing registration for {name} from {image_source}.")
        face_embedding, box = detect_and_encode_face(image)

        # Clean up temporary file as soon as image is processed (or fails)
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
            return render_template('register.html'), 400

        # --- Store User in Database ---
        try:
            india_tz = pytz.timezone('Asia/Kolkata')
            # Convert NumPy array embedding to list for MongoDB storage
            embedding_list = face_embedding.tolist()
            user_data = {
                'name': name,
                'role': role, # Store the role
                'face_embedding': embedding_list,
                'registered_at': datetime.now(india_tz)
            }
            # Include branch only if role is student or staff
            if role in ['student', 'staff']:
                user_data['branch'] = branch
            else:
                 user_data['branch'] = None # Or omit, depending on preference

            # Using 'students' collection name as per original code (consider renaming later)
            mongo.db.students.insert_one(user_data)
            logging.info(f"Successfully registered user: {name} with role: {role}")
            flash(f"User '{name}' ({role}) registered successfully!", "success")
            load_known_faces() # Refresh the in-memory cache
            return redirect(url_for('view_students')) # Redirect to the student list

        except Exception as e:
            logging.error(f"Database error during registration for {name}: {e}")
            flash("Database error occurred during registration.", "error")
            # No need to remove file here again, should have been removed earlier
            return render_template('register.html'), 500

    # GET request: just render the registration form
    return render_template('register.html')


@app.route('/view_students', methods=['GET'])
def view_students():
    """ Displays the list of registered students with search, role filter, branch filter, and sort (Admin only) """
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    # --- Removed Temporary Simplification ---

    try:
        search_query = request.args.get('search', '')
        role_filter = request.args.get('role_filter', '') # Get the role filter
        branch_filter = request.args.get('branch_filter', '') # Get the branch filter from dropdown
        sort_by = request.args.get('sort_by', 'name') # Default sort by name
        sort_order = request.args.get('sort_order', 'asc') # Default sort ascending

        # --- Fetch Distinct Branches for Dropdown ---
        try:
            # Get unique non-null/non-empty branch values from users with roles 'student' or 'staff'
            distinct_branches = mongo.db.students.distinct('branch', {'branch': {'$nin': [None, '']}, 'role': {'$in': ['student', 'staff']}})
            # Sort branches alphabetically for the dropdown
            distinct_branches.sort()
        except Exception as db_err:
            logging.error(f"Error fetching distinct branches: {db_err}")
            flash("Error loading branch filter options.", "error")
            distinct_branches = [] # Provide empty list on error

        # --- Build MongoDB Query ---
        query_conditions = []
        if search_query:
            # Case-insensitive search on name or branch
            regex = {'$regex': search_query, '$options': 'i'}
            query_conditions.append({'$or': [{'name': regex}, {'branch': regex}]})

        if role_filter:
            # Add role filter condition
            query_conditions.append({'role': role_filter})

        if branch_filter:
            # Add branch filter condition
            query_conditions.append({'branch': branch_filter})

        # Combine conditions using $and if multiple exist
        query = {}
        if query_conditions:
            query['$and'] = query_conditions
        # If no conditions, query remains empty {} which fetches all

        # --- Determine Sort Order ---
        mongo_sort_order = 1 if sort_order == 'asc' else -1
        # Valid sort fields
        valid_sort_fields = ['name', 'role', 'registered_at', 'branch'] # Added 'branch'
        if sort_by not in valid_sort_fields:
            sort_by = 'name' # Default to name if invalid field provided

        # --- Fetch Data ---
        # Fetch 'role' field as well
        students_cursor = mongo.db.students.find(query, {"name": 1, "branch": 1, "role": 1, "registered_at": 1, "_id": 1}) \
                                          .sort(sort_by, mongo_sort_order)
        students_list = list(students_cursor)

        # Format datetime and add string representation of ObjectId
        india_tz = pytz.timezone('Asia/Kolkata')
        for student in students_list:
            if 'registered_at' in student and isinstance(student['registered_at'], datetime):
                student['registered_at_str'] = student['registered_at'].astimezone(india_tz).strftime('%Y-%m-%d %H:%M:%S')
            else:
                student['registered_at_str'] = "N/A"
            # Convert ObjectId to string for URL generation in template
            student['_id_str'] = str(student['_id'])

        # --- DEBUG: Log distinct branches ---
        logging.info(f"Distinct branches being passed to template: {distinct_branches}")
        # --- END DEBUG ---

        return render_template('view_students.html',
                               students=students_list,
                               search_query=search_query,
                               role_filter=role_filter,
                               branch_filter=branch_filter, # Pass branch_filter to template
                               distinct_branches=distinct_branches, # Pass branches for dropdown
                               sort_by=sort_by,
                                sort_order=sort_order)

    except Exception as e:
        # Log the full traceback for detailed debugging
        logging.exception(f"Error fetching students list for view:") # Use logging.exception to include traceback
        # Attempt to fetch branches even on general error for consistent UI
        try:
            distinct_branches = mongo.db.students.distinct('branch', {'branch': {'$nin': [None, '']}, 'role': {'$in': ['student', 'staff']}})
            distinct_branches.sort()
        except Exception as db_err:
             logging.error(f"Error fetching distinct branches during exception handling: {db_err}")
             distinct_branches = [] # Fallback if branch fetch also fails

        flash("Error loading student data. Please check logs or contact support.", "error") # More informative flash
        # Pass empty list and current params back to template on error, but return 200 OK
        # Returning 200 might prevent session issues triggered by 500 errors.
        return render_template('view_students.html',
                               students=[],
                               search_query=request.args.get('search', ''),
                               role_filter=request.args.get('role_filter', ''),
                               branch_filter=request.args.get('branch_filter', ''), # Pass branch_filter even on error
                               distinct_branches=distinct_branches, # Pass branches even on error
                               sort_by=request.args.get('sort_by', 'name'),
                               sort_order=request.args.get('sort_order', 'asc')), 200 # Return 200 OK


@app.route('/delete_student/<student_id>', methods=['POST'])
def delete_student(student_id):
    """ Handles deleting a student (Admin only) """
    # No explicit @login_required needed due to restrict_access middleware

    try:
        oid = ObjectId(student_id)
    except Exception:
        flash("Invalid student ID format.", "error")
        return redirect(url_for('view_students'))

    try:
        # Find the student first to get their name for logging/cache removal
        student_to_delete = mongo.db.students.find_one({'_id': oid}, {'name': 1})
        if not student_to_delete:
            flash("Student not found for deletion.", "error")
            return redirect(url_for('view_students')), 404

        student_name = student_to_delete.get('name', 'Unknown') # Get name before deleting

        # Delete the student
        result = mongo.db.students.delete_one({'_id': oid})

        if result.deleted_count > 0:
            logging.info(f"Successfully deleted student: {student_name} (ID: {student_id})")
            flash(f"Student '{student_name}' deleted successfully.", "success")
            load_known_faces() # Update the cache

            # Optionally remove associated sightings from seen_log
            try:
                log_result = mongo.db.seen_log.delete_many({'name': student_name, 'type': 'known_sighting'})
                logging.info(f"Removed {log_result.deleted_count} sighting logs for deleted student {student_name}.")
            except Exception as log_e:
                logging.error(f"Error removing sighting logs for {student_name}: {log_e}")

        else:
            # This case should ideally not happen if find_one succeeded, but good to handle
            logging.warning(f"Student with ID {student_id} found but deletion failed.")
            flash("Student found but could not be deleted.", "error")

    except Exception as e:
        logging.error(f"Error deleting student {student_id}: {e}")
        flash("An error occurred while trying to delete the student.", "error")

    return redirect(url_for('view_students'))


@app.route('/configure', methods=['GET', 'POST'])
def configure():
    """ Handles configuration settings (Admin only) """
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    config = load_config()

    if request.method == 'POST':
        # --- Unknown Face Log Interval Handling ---
        unknown_interval_str = request.form.get('unknown_log_interval_seconds')
        unknown_interval_flash = {'message': "", 'category': "info"} # Default category

        if unknown_interval_str is not None: # Check if the field exists
            try:
                unknown_interval_val = int(unknown_interval_str)
                if unknown_interval_val >= 1: # Minimum interval of 1 second
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
                unknown_interval_flash['category'] = "warning"
                logging.warning(f"Invalid unknown_log_interval_seconds input received: '{unknown_interval_str}'. Not saved.")
            except Exception as e:
                logging.error(f"Error processing unknown_log_interval_seconds '{unknown_interval_str}': {e}")
                unknown_interval_flash['message'] = "An unexpected error occurred saving the unknown face log interval."
                unknown_interval_flash['category'] = "error"
        else:
             # This case might occur if the field name is wrong in the HTML or request is malformed
             unknown_interval_flash['message'] = "Unknown face log interval field was missing from the request. Value not saved."
             unknown_interval_flash['category'] = "warning"
             logging.warning("unknown_log_interval_seconds field missing from POST request.")

        # --- Stop Time Handling ---
        enable_stop_time = request.form.get('enable_stop_time') == 'on'
        stop_time_str = request.form.get('stop_time')
        config['stop_time_enabled'] = enable_stop_time

        if enable_stop_time:
            # Validate time only if the feature is enabled
            if stop_time_str and len(stop_time_str) == 5 and stop_time_str[2] == ':':
                try:
                    # Further validation by trying to parse
                    datetime.strptime(stop_time_str, '%H:%M')
                    config['stop_time'] = stop_time_str
                    flash_message = f"Live feed stop time enabled and set to {stop_time_str}."
                    flash_category = "success"
                except ValueError:
                    # Keep enabled flag, but don't save invalid time, maybe keep old time?
                    # For simplicity, let's just report error and not save the invalid time.
                    # config['stop_time'] = config.get('stop_time', '') # Keep old time if invalid new one
                    flash_message = "Stop time enabled, but the provided time format was invalid. Please use HH:MM (24-hour). Previous time (if any) kept."
                    flash_category = "warning"
                except Exception as e:
                     logging.error(f"Error processing/saving stop time: {e}")
                     flash_message = "An error occurred while saving the time."
                     flash_category = "error"
            else:
                # Enabled, but time string is missing or malformed
                # config['stop_time'] = config.get('stop_time', '') # Keep old time
                flash_message = "Stop time enabled, but no valid time was provided. Please enter a time in HH:MM format."
                flash_category = "warning"
        else:
            # Feature disabled, no need to validate time. Optionally clear it.
            # config['stop_time'] = '' # Optionally clear time when disabled
            flash_message = "Automatic live feed stop time disabled."
            flash_category = "success"
        # Store stop time message details
        stop_time_flash = {'message': flash_message, 'category': flash_category}


        # --- Unknown Face Log Interval Handling ---
        unknown_interval_str = request.form.get('unknown_log_interval_seconds')
        unknown_interval_flash = {'message': "", 'category': "info"} # Default category

        if unknown_interval_str is not None: # Check if the field exists
            try:
                unknown_interval_val = int(unknown_interval_str)
                if unknown_interval_val >= 1: # Minimum interval of 1 second
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
                unknown_interval_flash['category'] = "warning"
                logging.warning(f"Invalid unknown_log_interval_seconds input received: '{unknown_interval_str}'. Not saved.")
            except Exception as e:
                logging.error(f"Error processing unknown_log_interval_seconds '{unknown_interval_str}': {e}")
                unknown_interval_flash['message'] = "An unexpected error occurred saving the unknown face log interval."
                unknown_interval_flash['category'] = "error"
        else:
             unknown_interval_flash['message'] = "Unknown face log interval field was missing from the request. Value not saved."
             unknown_interval_flash['category'] = "warning"
             logging.warning("unknown_log_interval_seconds field missing from POST request.")


        # --- Save Configuration and Flash Messages ---
        try:
            save_config(config)
            # Flash messages after saving is confirmed successful
            if stop_time_flash.get('message'):
                 flash(stop_time_flash['message'], stop_time_flash['category'])
            if unknown_interval_flash.get('message'):
                 flash(unknown_interval_flash['message'], unknown_interval_flash['category'])
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            flash("Failed to save one or more configuration settings.", "error")
            # Still redirect even on save error
            return redirect(url_for('configure'))

        # Redirect back to configure page after POST to show updated values/flash messages
        return redirect(url_for('configure'))

    # GET request
    # Load interval for GET request, ensuring it's an int and >= 1 for display
    try:
        unknown_log_interval_seconds = int(config.get('unknown_log_interval_seconds', 60)) # Default 60s
        if unknown_log_interval_seconds < 1:
            unknown_log_interval_seconds = 60 # Default to 60 if stored value is invalid
    except (ValueError, TypeError):
        unknown_log_interval_seconds = 60 # Default to 60 if stored value is not an int

    current_stop_time = config.get('stop_time', '')
    stop_time_enabled = config.get('stop_time_enabled', False)

    return render_template('configure.html',
                           current_stop_time=current_stop_time,
                           stop_time_enabled=stop_time_enabled,
                           unknown_log_interval_seconds=unknown_log_interval_seconds) # Pass validated value to template


# Removed upload_photo route as registration handles uploads now.
# If a separate generic admin upload is needed, it could be added back.

@app.route('/login', methods=['GET', 'POST'])
def login():
    """ Handles admin login """
    # If already logged in as admin, redirect to index immediately
    if session.get('user_type') == 'admin':
        return redirect(url_for('index'))

    if request.method == 'POST':
        password = request.form.get('admin_password') # Get password from the specific admin field

        # Simple password check for admin
        # Use environment variables for real passwords!
        admin_pass = os.environ.get("ADMIN_PASSWORD", "0000") # Default for testing
        if password == admin_pass:
            session['user_type'] = 'admin'
            session['user_name'] = 'Admin' # Set admin name
            logging.info("Admin login successful.")
            flash("Admin login successful.", "success")
            return redirect(url_for('index')) # Admin dashboard
        else:
            logging.warning("Invalid admin password attempt.")
            flash("Invalid admin credentials.", "error")
            # Re-render login page on failed attempt
            return render_template('login.html'), 401 # Unauthorized

    # GET request: Show login page (only if not already logged in)
    return render_template('login.html')


@app.route('/logout')
def logout():
    """ Clears the session and logs the user out """
    user_name = session.get('user_name', 'User') # Get name before popping
    session.pop('user_type', None)
    session.pop('user_name', None)
    logging.info(f"User '{user_name}' logged out.")
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))


@app.before_request
def restrict_access():
    """ Middleware to check login status before accessing protected routes """
    # Allow access to login, logout, and static files without being logged in
    if request.endpoint in ['login', 'static', 'logout']:
        return None

    # If no user type in session, redirect to login
    if 'user_type' not in session:
        flash("Please log in to access this page.", "info")
        return redirect(url_for('login'))

    # If user is logged in as admin, allow access
    if session.get('user_type') == 'admin':
        return None # Admin allowed

    # If not logged in as admin, redirect to login
    flash("Admin access required. Please log in.", "warning")
    return redirect(url_for('login'))


# --- Seen Log Routes (Kept) ---
@app.route('/seen_log')
def seen_log_view():
    """ Displays the log of all face sightings, grouped by person (Admin only) """
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))
    try:
        # Fetch logs, sort by timestamp descending
        log_records_cursor = mongo.db.seen_log.find().sort('timestamp', -1) # Fetch cursor first
        log_records = list(log_records_cursor) # Convert to list
        logging.debug(f"Fetched {len(log_records)} seen log records.")

        # Group logs by person
        grouped_logs = defaultdict(list)
        unknown_logs = [] # Separate list for unknown sightings

        india_tz = pytz.timezone('Asia/Kolkata')
        for record in log_records:
            # Safely get timestamp and format it
            ts = record.get('timestamp')
            if isinstance(ts, datetime):
                record['timestamp_str'] = ts.astimezone(india_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                record['timestamp_str'] = "Invalid Date"

            # Group by name if known, otherwise add to unknown list
            if record.get('type') == 'known_sighting':
                name = record.get('name')
                if name:
                    grouped_logs[name].append(record)
                else:
                    # Handle known sightings with missing names (data error)
                    record['name'] = 'Unknown (Error)'
                    unknown_logs.append(record) # Add to unknowns for now
            elif record.get('type') == 'unknown_sighting':
                 # Add unknown sightings to their own list
                 unknown_logs.append(record)
            # Add handling for other potential log types if necessary

        # Convert defaultdict to regular dict for template compatibility if needed
        grouped_logs_dict = dict(grouped_logs)

        # Sort logs within each person's group by timestamp descending (already sorted by fetch)
        # If sorting needed within groups:
        # for name in grouped_logs_dict:
        #     grouped_logs_dict[name].sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        # unknown_logs.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)


        return render_template('seen_log.html', grouped_logs=grouped_logs_dict, unknown_logs=unknown_logs)
    except Exception as e:
        logging.error(f"Error fetching or grouping seen log records: {e}")
        flash("Error loading seen log data.", "error")
        # Pass empty structures on error
        return render_template('seen_log.html', grouped_logs={}, unknown_logs=[]), 500


@app.route('/reset_seen_log', methods=['POST'])
def reset_seen_log():
    """ Clears all entries from the seen_log collection (Admin only) """
    if session.get('user_type') != 'admin':
        logging.warning("Unauthorized attempt to reset seen log.")
        flash("Unauthorized action.", "error")
        return redirect(url_for('login'))

    try:
        seen_log_collection = mongo.db.seen_log
        result = seen_log_collection.delete_many({}) # Delete all documents
        deleted_count = result.deleted_count
        logging.info(f"Seen log reset requested by admin. {deleted_count} records deleted.")
        # Removed flash message here
    except Exception as e:
        logging.error(f"Error resetting seen log: {e}")
        flash("An error occurred while resetting the seen log.", "error")

    # Redirect back to the seen log page regardless of success or failure
    return redirect(url_for('seen_log_view'))
@app.route('/get_attendance')
def get_attendance():
    """ Endpoint to calculate and return attendance (first/last seen) for known users for the current day. """
    if session.get('user_type') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now_india = datetime.now(india_tz)
        start_of_day = india_tz.localize(datetime.combine(now_india.date(), time.min))
        end_of_day = india_tz.localize(datetime.combine(now_india.date(), time.max))

        logging.info(f"Fetching attendance logs between {start_of_day} and {end_of_day}")

        # Query for known sightings within the current day in India timezone
        log_records_cursor = mongo.db.seen_log.find({
            'type': 'known_sighting',
            'timestamp': {
                '$gte': start_of_day,
                '$lte': end_of_day
            }
        }).sort('timestamp', 1) # Sort ascending to easily find first/last

        attendance_data = defaultdict(lambda: {'arriving': None, 'leaving': None, 'timestamps': []})

        for record in log_records_cursor:
            name = record.get('name')
            timestamp = record.get('timestamp')
            if name and timestamp:
                # Ensure timestamp is timezone-aware (should be from logging)
                if timestamp.tzinfo is None:
                     timestamp = india_tz.localize(timestamp) # Add timezone if missing (fallback)
                else:
                     timestamp = timestamp.astimezone(india_tz) # Convert to India TZ just in case

                attendance_data[name]['timestamps'].append(timestamp)

        # Process the collected timestamps for each person
        final_attendance = {}
        for name, data in attendance_data.items():
            if data['timestamps']:
                timestamps = sorted(data['timestamps']) # Ensure sorted
                arriving_time = timestamps[0]
                leaving_time = timestamps[-1]
                final_attendance[name] = {
                    'name': name, # Include name for easier frontend processing
                    'arriving_time': arriving_time.strftime('%Y-%m-%d %H:%M:%S'),
                    # Corrected: Always format the leaving_time, even if it's the same as arriving_time
                    'leaving_time': leaving_time.strftime('%Y-%m-%d %H:%M:%S')
                }

        logging.info(f"Processed attendance for {len(final_attendance)} individuals.")
        # Convert final_attendance dict to a list of objects for easier iteration in JS
        return jsonify(list(final_attendance.values()))

    except Exception as e:
        logging.error(f"Error calculating attendance: {e}")
        return jsonify({"error": "Failed to calculate attendance"}), 500

@app.route('/attendance')
def attendance_page():
    """ Renders the dedicated attendance page. """
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now_india = datetime.now(india_tz)
        start_of_day = india_tz.localize(datetime.combine(now_india.date(), time.min))
        end_of_day = india_tz.localize(datetime.combine(now_india.date(), time.max))

        logging.info(f"Fetching attendance logs for attendance page between {start_of_day} and {end_of_day}")

        # Query for known sightings within the current day in India timezone
        log_records_cursor = mongo.db.seen_log.find({
            'type': 'known_sighting',
            'timestamp': {
                '$gte': start_of_day,
                '$lte': end_of_day
            }
        }).sort('timestamp', 1) # Sort ascending to easily find first/last

        attendance_data = defaultdict(lambda: {'timestamps': []})
        user_details = {} # Store branch/role info

        # Pre-fetch user details to avoid multiple DB calls inside the loop
        known_users = list(mongo.db.students.find({}, {"name": 1, "branch": 1, "role": 1}))
        for user in known_users:
            user_details[user['name']] = {'branch': user.get('branch', 'N/A'), 'role': user.get('role', 'N/A')}


        for record in log_records_cursor:
            name = record.get('name')
            timestamp = record.get('timestamp')
            if name and timestamp:
                # Ensure timestamp is timezone-aware
                if timestamp.tzinfo is None:
                     timestamp = india_tz.localize(timestamp)
                else:
                     timestamp = timestamp.astimezone(india_tz)

                attendance_data[name]['timestamps'].append(timestamp)

        # Process the collected timestamps for each person
        final_attendance = []
        for name, data in attendance_data.items():
            if data['timestamps']:
                timestamps = sorted(data['timestamps']) # Ensure sorted
                arriving_time = timestamps[0]
                leaving_time = timestamps[-1] if len(timestamps) > 1 else arriving_time # Handle single sighting case
                details = user_details.get(name, {'branch': 'N/A', 'role': 'N/A'}) # Get details or default
                final_attendance.append({
                    'name': name,
                    'branch': details['branch'],
                    'role': details['role'],
                    'arriving_time': arriving_time.strftime('%H:%M:%S'), # Format as HH:MM:SS
                    'leaving_time': leaving_time.strftime('%H:%M:%S'), # Format as HH:MM:SS
                    'date': arriving_time.strftime('%Y-%m-%d') # Add date
                })

        logging.info(f"Processed attendance for {len(final_attendance)} individuals for attendance page.")
        # Sort by name for consistent display
        final_attendance.sort(key=lambda x: x['name'])

        return render_template('attendance.html', attendance_records=final_attendance)

    except Exception as e:
        logging.error(f"Error generating attendance page: {e}")
        flash("Error loading attendance data.", "error")
        return render_template('attendance.html', attendance_records=[]), 500

@app.route('/export_attendance_excel')
def export_attendance_excel():
    """ Exports the current day's attendance data to an Excel file. """
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))

    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now_india = datetime.now(india_tz)
        start_of_day = india_tz.localize(datetime.combine(now_india.date(), time.min))
        end_of_day = india_tz.localize(datetime.combine(now_india.date(), time.max))
        current_date_str = now_india.strftime('%Y-%m-%d')

        logging.info(f"Fetching attendance logs for Excel export between {start_of_day} and {end_of_day}")

        # --- Fetch Data (Similar to attendance_page) ---
        log_records_cursor = mongo.db.seen_log.find({
            'type': 'known_sighting',
            'timestamp': {
                '$gte': start_of_day,
                '$lte': end_of_day
            }
        }).sort('timestamp', 1)

        attendance_data = defaultdict(lambda: {'timestamps': []})
        user_details = {}
        known_users = list(mongo.db.students.find({}, {"name": 1, "branch": 1, "role": 1}))
        for user in known_users:
            user_details[user['name']] = {'branch': user.get('branch', 'N/A'), 'role': user.get('role', 'N/A')}

        for record in log_records_cursor:
            name = record.get('name')
            timestamp = record.get('timestamp')
            if name and timestamp:
                if timestamp.tzinfo is None: timestamp = india_tz.localize(timestamp)
                else: timestamp = timestamp.astimezone(india_tz)
                attendance_data[name]['timestamps'].append(timestamp)

        final_attendance = []
        for name, data in attendance_data.items():
            if data['timestamps']:
                timestamps = sorted(data['timestamps'])
                arriving_time = timestamps[0]
                leaving_time = timestamps[-1] if len(timestamps) > 1 else arriving_time
                details = user_details.get(name, {'branch': 'N/A', 'role': 'N/A'})
                final_attendance.append({
                    'name': name,
                    'branch': details['branch'],
                    'role': details['role'],
                    'arriving_time': arriving_time.strftime('%H:%M:%S'),
                    'leaving_time': leaving_time.strftime('%H:%M:%S'),
                    'date': arriving_time.strftime('%Y-%m-%d')
                })
        final_attendance.sort(key=lambda x: x['name'])
        # --- End Fetch Data ---

        # --- Create Excel Workbook ---
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = f"Attendance {current_date_str}"

        # Add Headers
        headers = ["Name", "Branch", "Role", "Arriving Time", "Leaving Time", "Date"]
        sheet.append(headers)

        # Add Data Rows
        for record in final_attendance:
            sheet.append([
                record['name'],
                record['branch'],
                record['role'],
                record['arriving_time'],
                record['leaving_time'],
                record['date']
            ])

        # --- Save to Memory ---
        excel_stream = BytesIO()
        workbook.save(excel_stream)
        excel_stream.seek(0) # Rewind the stream to the beginning

        logging.info(f"Generated Excel export for {len(final_attendance)} records for date {current_date_str}.")

        # --- Return Response ---
        return Response(
            excel_stream,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment;filename=attendance_{current_date_str}.xlsx'}
        )

    except Exception as e:
        logging.error(f"Error generating Excel export: {e}")
        flash("Error generating Excel export.", "error")
        # Redirect back to attendance page on error
        return redirect(url_for('attendance_page'))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Set debug=False for production, threaded=True can help with multiple requests
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
