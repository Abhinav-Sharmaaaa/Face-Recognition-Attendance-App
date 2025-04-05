from flask import Flask, render_template, request, redirect, url_for, jsonify, session, Response, flash
import os
from datetime import datetime, timedelta # Added timedelta
import pytz # Kept for seen_log timestamps
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


# --- Known Face Data Handling (Unchanged) ---
known_face_data = {} # Cache: name -> {'embedding': np.array, 'branch': str}
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
        # Fetch only necessary fields
        # Using 'students' collection as per original code
        users_cursor = mongo.db.students.find({}, {"name": 1, "branch": 1, "face_embedding": 1})
        count = 0
        for user_doc in users_cursor:
            # Check if essential data exists
            if 'face_embedding' in user_doc and user_doc['face_embedding'] and 'name' in user_doc:
                try:
                    # Convert embedding from list (stored in MongoDB) to NumPy array
                    embedding_np = np.array(user_doc['face_embedding']).astype(np.float32)
                    # Validate shape (SFace generates 128-dim features)
                    if embedding_np.shape == (128,):
                        known_face_data[user_doc['name']] = {
                            'embedding': embedding_np,
                            'branch': user_doc.get('branch', 'N/A') # Handle missing branch
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
    """ Generates frames from webcam, performs face recognition, and logs sightings. """
    cap = cv2.VideoCapture(0) # Use camera 0
    if not cap.isOpened():
        logging.error("Cannot open camera 0")
        return

    frame_count = 0
    process_every_n_frames = 3 # Process every 3rd frame to save resources
    last_known_faces = {} # Store last detected faces for smoother display
    india_tz = pytz.timezone('Asia/Kolkata') # Timezone for logging
    seen_log_collection = mongo.db.seen_log # Collection for logging ALL sightings
    last_log_times = {} # Dictionary to track last log time for each person
    log_interval = timedelta(minutes=1) # Set the minimum interval between logs

    while True:
        success, frame = cap.read()
        if not success:
            logging.warning("Failed to read frame from camera.")
            break

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

                # --- Logging Sighting Logic (Kept as it's not attendance) ---
                if box: # Only log if a face box was detected
                    log_entry = {
                        'timestamp': current_time_india,
                        'status_at_log': status_text # Log the status determined above
                    }
                    if recognized_user_name: # Known face detected
                        now = current_time_india # Use the already fetched timestamp
                        last_logged = last_log_times.get(recognized_user_name)

                        # Check if the person was logged before and if the interval has passed
                        if last_logged is None or (now - last_logged) >= log_interval:
                            log_entry['type'] = 'known_sighting'
                            log_entry['name'] = recognized_user_name
                            try:
                                seen_log_collection.insert_one(log_entry)
                                last_log_times[recognized_user_name] = now # Update last log time
                                logging.info(f"Logged sighting of known face: {recognized_user_name} (Interval passed or first time)")
                            except Exception as db_err:
                                logging.error(f"Error inserting known sighting log for {recognized_user_name}: {db_err}")
                        # else: # Interval not passed, do not log again yet
                        #    logging.debug(f"Skipping log for {recognized_user_name}, interval not passed.")

                    elif status_text == "Unknown Face": # Unknown face detected
                        # Log unknown faces every time they are detected (no interval needed for unknowns)
                        log_entry['type'] = 'unknown_sighting'
                        # Optionally add cropped face image for unknown faces
                        try:
                            (x, y, w, h) = box
                            # Basic validation for cropping coordinates
                            if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= processing_frame.shape[1] and y + h <= processing_frame.shape[0]:
                                cropped_face = processing_frame[y:y+h, x:x+w]
                                if cropped_face.size > 0:
                                    _, buffer = cv2.imencode('.jpg', cropped_face)
                                    log_entry['face_image_base64'] = base64.b64encode(buffer).decode('utf-8')
                                else:
                                     logging.warning("Cropped unknown face is empty, logging without image.")
                            else:
                                 logging.warning(f"Invalid box coordinates for cropping unknown face: {box}, frame shape: {processing_frame.shape}")

                            # Insert log even if image cropping failed
                            seen_log_collection.insert_one(log_entry)
                            logging.info(f"Logged unknown face sighting {'with image' if 'face_image_base64' in log_entry else 'without image'}.")
                        except Exception as log_err:
                            logging.error(f"Error preparing or inserting unknown sighting log: {log_err}")
                    # else: # Low match or processing error - currently not logging these separately in seen_log
                    #    pass
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
        branch = request.form.get('branch') # Or department, group etc.
        photo_data = request.form.get('photo') # From webcam capture
        uploaded_file = request.files.get('uploadPhoto') # From file upload

        if not name or not branch:
            flash("Name and Branch/Department are required.", "error")
            return render_template('register.html'), 400 # Re-render form with error

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
            # Using 'students' collection name as per original code
            mongo.db.students.insert_one({
                'name': name,
                'branch': branch,
                'face_embedding': embedding_list,
                'registered_at': datetime.now(india_tz)
            })
            logging.info(f"Successfully registered user: {name}")
            flash(f"User '{name}' registered successfully!", "success")
            load_known_faces() # Refresh the in-memory cache
            return redirect(url_for('users_view')) # Redirect to the list of users

        except Exception as e:
            logging.error(f"Database error during registration for {name}: {e}")
            flash("Database error occurred during registration.", "error")
            # No need to remove file here again, should have been removed earlier
            return render_template('register.html'), 500

    # GET request: just render the registration form
    return render_template('register.html')


@app.route('/remove_user/<name>', methods=['POST']) # Renamed route and parameter
def remove_user(name): # Renamed function
    """ Removes a registered user (Admin only) """
    if session.get('user_type') != 'admin':
         flash("Unauthorized action.", "error")
         return "Unauthorized", 403

    try:
        # Decode the name from URL (might contain spaces etc.)
        decoded_name = urllib.parse.unquote(name)
        logging.info(f"Attempting to remove user: {decoded_name}")

        # Using 'students' collection name as per original code
        result = mongo.db.students.delete_one({'name': decoded_name})

        if result.deleted_count > 0:
            logging.info(f"Successfully removed user: {decoded_name}")
            flash(f"User '{decoded_name}' removed successfully.", "success")
            load_known_faces() # Update the cache
            # Also remove associated sightings from seen_log (Optional but good practice)
            try:
                log_result = mongo.db.seen_log.delete_many({'name': decoded_name, 'type': 'known_sighting'})
                logging.info(f"Removed {log_result.deleted_count} sighting logs for deleted user {decoded_name}.")
            except Exception as log_e:
                logging.error(f"Error removing sighting logs for {decoded_name}: {log_e}")

            return redirect(url_for('users_view')) # Redirect to the updated user list
        else:
            logging.warning(f"User not found for removal: {decoded_name}")
            flash(f"User '{decoded_name}' not found!", "error")
            return redirect(url_for('users_view')), 404 # Not found

    except Exception as e:
         logging.error(f"Error removing user {name}: {e}")
         flash("An error occurred while trying to remove the user.", "error")
         return redirect(url_for('users_view')), 500 # Internal server error


@app.route('/users', methods=['GET']) # Renamed route
def users_view(): # Renamed function
    """ Displays the list of registered users (Admin only) """
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))
    try:
        # Fetch users, projecting only necessary fields
        # Using 'students' collection name as per original code
        users_cursor = mongo.db.students.find({}, {"name": 1, "branch": 1, "registered_at": 1})
        users_list = list(users_cursor)

        # Format datetime for display
        for user in users_list:
             if 'registered_at' in user and isinstance(user['registered_at'], datetime):
                 # Assuming stored time is UTC or timezone-aware in DB
                 # Format it nicely
                 user['registered_at_str'] = user['registered_at'].strftime('%Y-%m-%d %H:%M:%S')
             else:
                 user['registered_at_str'] = "N/A" # Handle cases where date might be missing/invalid

        return render_template('users.html', users=users_list) # Needs users.html template

    except Exception as e:
        logging.error(f"Error fetching users list: {e}")
        flash("Error loading user data.", "error")
        return render_template('users.html', users=[]), 500 # Show empty list on error


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
    """ Displays the log of all face sightings (Admin only) """
    if session.get('user_type') != 'admin':
        flash("Unauthorized access.", "warning")
        return redirect(url_for('login'))
    try:
        # Fetch logs, sort by timestamp descending, limit for performance
        log_records = list(mongo.db.seen_log.find().sort('timestamp', -1).limit(500)) # Limit to last 500 entries
        logging.debug(f"Fetched {len(log_records)} seen log records.")

        # Format timestamp for display
        india_tz = pytz.timezone('Asia/Kolkata')
        for record in log_records:
             # Safely get timestamp and format it
             ts = record.get('timestamp')
             if isinstance(ts, datetime):
                  # Convert to local timezone if needed (assuming stored as UTC or naive)
                  record['timestamp_str'] = ts.astimezone(india_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
             else:
                  record['timestamp_str'] = "Invalid Date" # Handle bad data

             # Make user 'Unknown' if name is missing (for known_sightings with data issue)
             if record.get('type') == 'known_sighting' and not record.get('name'):
                 record['name'] = 'Unknown (Error)'

        return render_template('seen_log.html', logs=log_records) # Needs seen_log.html template
    except Exception as e:
        logging.error(f"Error fetching seen log records: {e}")
        flash("Error loading seen log data.", "error")
        return render_template('seen_log.html', logs=[]), 500 # Show empty on error


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
        flash(f"Seen log successfully reset. {deleted_count} records removed.", "success")
    except Exception as e:
        logging.error(f"Error resetting seen log: {e}")
        flash("An error occurred while resetting the seen log.", "error")

    return redirect(url_for('seen_log_view')) # Redirect back to the log view


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Set debug=False for production, threaded=True can help with multiple requests
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
