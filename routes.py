import os
import os
import cv2
import logging
import json # <-- Add json import
from flask import (
    Flask, render_template, Response, request, redirect, url_for, flash, jsonify, session, send_from_directory
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime

# Import utility functions and configurations
import config
import database
import face_utils
import attendance_utils

# Initialize Flask App (will be imported by app.py)
app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY=config.SECRET_KEY,
    SQLALCHEMY_DATABASE_URI=config.DATABASE_URI, # Use the SQLite URI from config
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    UPLOAD_FOLDER=config.UPLOAD_FOLDER
)

# Initialize Database with the app instance
database.init_db(app) # Use the init_db function from database.py

# Define path for the seen log file
SEEN_LOG_FILE = os.path.join(app.instance_path, 'seen_log.json') # Store in instance folder

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Ensure instance folder exists (for the seen log)
os.makedirs(app.instance_path, exist_ok=True)


# --- Helper function to read seen log ---
def get_seen_log():
    """Reads the seen log data from the JSON file."""
    try:
        if os.path.exists(SEEN_LOG_FILE):
            with open(SEEN_LOG_FILE, 'r') as f:
                # Read lines and parse JSON for each line, reverse for recent first
                log_data = [json.loads(line) for line in f if line.strip()]
                return sorted(log_data, key=lambda x: x.get('timestamp', ''), reverse=True)
        return []
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Error reading seen log file ({SEEN_LOG_FILE}): {e}")
        return [] # Return empty list on error

# --- Authentication ---
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD_HASH = generate_password_hash(os.environ.get('ADMIN_PASSWORD', 'password')) # Hash the default/env password

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Simple check against environment variables or defaults
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            attendance_utils.log_activity(f"Admin user '{username}' logged in.")
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            attendance_utils.log_activity(f"Failed login attempt for username '{username}'.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'Unknown user')
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    attendance_utils.log_activity(f"Admin user '{username}' logged out.")
    return redirect(url_for('login'))

# --- Core Routes ---
@app.route('/')
@login_required
def index():
    """Main dashboard page."""
    students = database.get_all_students()
    today_str = datetime.now().strftime("%Y-%m-%d")
    attendance_today = attendance_utils.get_attendance_records(filter_date=today_str)
    present_count = len(attendance_today.get(today_str, {}))
    total_students = len(students)
    absent_count = total_students - present_count
    return render_template('index.html',
                           students=students,
                           present_count=present_count,
                           absent_count=absent_count,
                           total_students=total_students)

@app.route('/live_feed')
@login_required
def live_feed():
    """Page to display the live camera feed for recognition."""
    return render_template('live_feed.html')

@app.route('/video_feed')
@login_required
def video_feed():
    """Provides the video stream with face recognition."""
    return Response(face_utils.generate_frames_with_recognition(attendance_utils.mark_attendance, attendance_utils.log_activity),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Student Management ---
@app.route('/students')
@login_required
def list_students():
    """Lists all registered students."""
    students = database.get_all_students()
    return render_template('students.html', students=students)

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register_student():
    """Handles student registration."""
    if request.method == 'POST':
        name = request.form['name']
        roll_number = request.form['roll_number']
        branch = request.form['branch']
        photo = request.files.get('photo') # Use .get to avoid KeyError if no file

        if not name or not roll_number or not branch:
            flash('Name, Roll Number, and Branch are required.', 'danger')
            return redirect(request.url)

        if database.get_student_by_roll(roll_number):
             flash(f'Student with Roll Number {roll_number} already exists.', 'warning')
             return redirect(request.url)

        filename = None
        photo_path = None
        if photo and photo.filename != '':
            if not face_utils.allowed_file(photo.filename):
                 flash('Invalid file type for photo. Allowed types: png, jpg, jpeg, gif', 'danger')
                 return redirect(request.url)

            filename = secure_filename(f"{roll_number}_{name}.{photo.filename.rsplit('.', 1)[1].lower()}")
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                photo.save(photo_path)
                # Optional: Validate if the uploaded image contains a face
                if not face_utils.image_contains_face(photo_path):
                    os.remove(photo_path) # Clean up invalid photo
                    flash('No face detected in the uploaded photo. Please upload a clear picture.', 'danger')
                    return redirect(request.url)
                logging.info(f"Photo saved for {name} at {photo_path}")
            except Exception as e:
                flash(f'Error saving photo: {e}', 'danger')
                logging.error(f"Error saving photo for {name}: {e}")
                return redirect(request.url)
        else:
            flash('Photo is required for registration.', 'danger')
            return redirect(request.url) # Stay on registration page if photo missing

        # Add student to database
        try:
            database.add_student(name, roll_number, branch, filename) # Store filename, not full path
            flash(f'Student {name} registered successfully!', 'success')
            attendance_utils.log_activity(f"Registered new student: {name} (Roll: {roll_number}, Branch: {branch})")
            # Update known faces cache after registration
            face_utils.load_known_faces(config.UPLOAD_FOLDER, database.get_all_students)
            return redirect(url_for('list_students'))
        except Exception as e:
            flash(f'Error registering student: {e}', 'danger')
            logging.error(f"Database error registering student {name}: {e}")
            # Clean up saved photo if DB registration failed
            if photo_path and os.path.exists(photo_path):
                os.remove(photo_path)
            return redirect(request.url)

    return render_template('register.html')


@app.route('/student/<int:student_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_student(student_id):
    """Handles editing student details."""
    student = database.get_student_by_id(student_id)
    if not student:
        flash('Student not found.', 'danger')
        return redirect(url_for('list_students'))

    if request.method == 'POST':
        original_name = student.name
        original_roll = student.roll_number
        original_branch = student.branch
        original_photo = student.photo_filename

        new_name = request.form['name']
        new_roll_number = request.form['roll_number']
        new_branch = request.form['branch']
        new_photo = request.files.get('photo')

        # Check if roll number changed and if the new one already exists
        if new_roll_number != original_roll and database.get_student_by_roll(new_roll_number):
            flash(f'Another student with Roll Number {new_roll_number} already exists.', 'warning')
            return render_template('edit_student.html', student=student)

        new_filename = original_photo
        photo_updated = False
        if new_photo and new_photo.filename != '':
            if not face_utils.allowed_file(new_photo.filename):
                flash('Invalid file type for photo. Allowed types: png, jpg, jpeg, gif', 'danger')
                return render_template('edit_student.html', student=student)

            # Create new filename, potentially based on new roll/name
            new_filename_base = secure_filename(f"{new_roll_number}_{new_name}")
            new_filename = f"{new_filename_base}.{new_photo.filename.rsplit('.', 1)[1].lower()}"
            new_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

            try:
                 # Validate new photo before saving
                temp_save_path = new_photo_path + ".tmp" # Save temporarily for validation
                new_photo.save(temp_save_path)
                if not face_utils.image_contains_face(temp_save_path):
                    os.remove(temp_save_path)
                    flash('No face detected in the new photo. Please upload a clear picture.', 'danger')
                    return render_template('edit_student.html', student=student)

                # If validation passes, rename temp file
                os.rename(temp_save_path, new_photo_path)

                # Remove old photo if filename changed and it exists
                if original_photo and new_filename != original_photo:
                    old_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], original_photo)
                    if os.path.exists(old_photo_path):
                        os.remove(old_photo_path)
                        logging.info(f"Removed old photo: {old_photo_path}")
                photo_updated = True
                logging.info(f"Updated photo saved for {new_name} at {new_photo_path}")
            except Exception as e:
                flash(f'Error saving new photo: {e}', 'danger')
                logging.error(f"Error saving updated photo for {new_name}: {e}")
                # Clean up temp file if error occurred after saving it
                if os.path.exists(temp_save_path):
                    os.remove(temp_save_path)
                return render_template('edit_student.html', student=student)

        # Update database
        try:
            database.update_student(student_id, new_name, new_roll_number, new_branch, new_filename)
            flash(f'Student {new_name} updated successfully!', 'success')
            attendance_utils.log_activity(f"Updated student details for ID {student_id} (Name: {original_name} -> {new_name}, Roll: {original_roll} -> {new_roll_number}, Branch: {original_branch} -> {new_branch}, Photo Updated: {photo_updated})")
             # Update known faces cache if photo was changed or name changed
            if photo_updated or new_name != original_name:
                 face_utils.load_known_faces(config.UPLOAD_FOLDER, database.get_all_students)
            return redirect(url_for('list_students'))
        except Exception as e:
            flash(f'Error updating student: {e}', 'danger')
            logging.error(f"Database error updating student ID {student_id}: {e}")
            # If DB update fails but photo was saved, attempt to revert photo change? (Complex)
            # For simplicity, we might leave the new photo saved.
            return render_template('edit_student.html', student=student)

    return render_template('edit_student.html', student=student)


@app.route('/student/<int:student_id>/delete', methods=['POST'])
@login_required
def delete_student(student_id):
    """Handles deleting a student."""
    student = database.get_student_by_id(student_id)
    if not student:
        flash('Student not found.', 'danger')
        return redirect(url_for('list_students'))

    try:
        student_name = student.name
        student_roll = student.roll_number
        photo_filename = student.photo_filename

        # Delete from database first
        database.delete_student(student_id)

        # Delete associated photo file
        if photo_filename:
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
            if os.path.exists(photo_path):
                os.remove(photo_path)
                logging.info(f"Deleted photo file: {photo_path}")

        flash(f'Student {student_name} deleted successfully.', 'success')
        attendance_utils.log_activity(f"Deleted student: {student_name} (Roll: {student_roll})")
        # Update known faces cache after deletion
        face_utils.load_known_faces(config.UPLOAD_FOLDER, database.get_all_students)
    except Exception as e:
        flash(f'Error deleting student: {e}', 'danger')
        logging.error(f"Error deleting student ID {student_id}: {e}")

    return redirect(url_for('list_students'))

# --- Attendance & Logs ---
@app.route('/attendance')
@login_required
def view_attendance():
    """Displays attendance records, allows filtering."""
    filter_date = request.args.get('date') # Get date from query param
    filter_branch = request.args.get('branch') # Get branch from query param

    # Validate date format if provided
    if filter_date:
        try:
            datetime.strptime(filter_date, '%Y-%m-%d')
        except ValueError:
            flash("Invalid date format. Please use YYYY-MM-DD.", "warning")
            filter_date = None # Reset to show all if format is wrong

    attendance_records = attendance_utils.get_attendance_records(filter_date=filter_date, filter_branch=filter_branch)
    branches = database.get_distinct_branches() # Get available branches for filter dropdown

    return render_template('attendance.html',
                           attendance_records=attendance_records,
                           branches=branches,
                           selected_date=filter_date,
                           selected_branch=filter_branch)

@app.route('/activity_log')
@login_required
def view_activity_log():
    """Displays the activity log."""
    log_entries = attendance_utils.get_activity_log()
    return render_template('activity_log.html', log_entries=log_entries)

@app.route('/seen_log')
@login_required
def view_seen_log():
    """Displays the log of when each person was seen."""
    seen_log_entries = get_seen_log()
    return render_template('seen_log.html', seen_log_entries=seen_log_entries)

# --- Serving Uploaded Files ---
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    """Serves uploaded files (student photos)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- Status Pages for Attendance Marking ---
# These routes are redirected to from mark_attendance via JavaScript in live_feed.html
@app.route('/present/<student_name>')
@login_required
def present_page(student_name):
    return render_template('present.html', student_name=student_name)

@app.route('/already_marked/<student_name>')
@login_required
def already_marked_page(student_name):
    return render_template('already_marked.html', student_name=student_name)

@app.route('/holiday')
@login_required
def holiday_page():
    return render_template('holiday.html')

# --- Error Handling ---
@app.errorhandler(404)
def page_not_found(e):
    attendance_utils.log_activity(f"404 Not Found error for path: {request.path}")
    return render_template('404.html'), 404 # Assuming you have a 404.html template

@app.errorhandler(500)
def internal_server_error(e):
    attendance_utils.log_activity(f"500 Internal Server Error: {e} for path: {request.path}")
    logging.error(f"500 Internal Server Error: {e}", exc_info=True) # Log the full traceback
    return render_template('500.html'), 500 # Assuming you have a 500.html template

# --- Initial Load ---
# Load known faces on startup after app context is available
with app.app_context():
    face_utils.load_known_faces(config.UPLOAD_FOLDER, database.get_all_students)
