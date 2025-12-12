import sqlite3
import os
import openai
import base64
import requests
from flask import Flask, request, jsonify
import uuid
from PIL import Image

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
db_path = "book_dataset.db"
upload_folder_crops = "./processed_images"
upload_folder_whole = "./whole_images"
os.makedirs(upload_folder_crops, exist_ok=True)
os.makedirs(upload_folder_whole, exist_ok=True)

openai.api_key = ""

# --- Database Functions ---
def create_db_if_not_exists():
    """
    Creates or updates the SQLite database and tables if they don't exist.
    This function is idempotent and safe to run on every startup.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Table for individual book crops, linked to a whole image
    c.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            author TEXT,
            subtitle TEXT,
            image_name TEXT NOT NULL,
            whole_image_id TEXT 
        )
    """)

    # Table for the whole captured images, including the detected bounding boxes
    c.execute("""
        CREATE TABLE IF NOT EXISTS whole_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            whole_image_id TEXT NOT NULL UNIQUE,
            image_name TEXT NOT NULL,
            bounding_boxes_json TEXT
        )
    """)
    
    try:
        c.execute("ALTER TABLE books ADD COLUMN whole_image_id TEXT")
        print("Added 'whole_image_id' column to 'books' table.")
    except sqlite3.OperationalError:
        pass
    
    try:
        c.execute("ALTER TABLE whole_images ADD COLUMN bounding_boxes_json TEXT")
        print("Added 'bounding_boxes_json' column to 'whole_images' table.")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

def insert_book_data(title, author, subtitle, image_name, whole_image_id):
    """Inserts the extracted book crop data into the 'books' table."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("INSERT INTO books (title, author, subtitle, image_name, whole_image_id) VALUES (?, ?, ?, ?, ?)",
              (title, author, subtitle, image_name, whole_image_id))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id

def insert_whole_image(whole_image_id, image_name, bounding_boxes_json):
    """Inserts a record for the whole image, including its bounding box data."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    new_id = None
    try:
        c.execute("INSERT INTO whole_images (whole_image_id, image_name, bounding_boxes_json) VALUES (?, ?, ?)",
                  (whole_image_id, image_name, bounding_boxes_json))
        conn.commit()
        new_id = c.lastrowid
    except sqlite3.IntegrityError:
        # This handles the rare case where the same UUID is sent twice.
        print(f"Warning: whole_image_id '{whole_image_id}' already exists. Record not inserted again.")
        c.execute("SELECT id FROM whole_images WHERE whole_image_id = ?", (whole_image_id,))
        result = c.fetchone()
        if result:
            new_id = result[0]
    finally:
        conn.close()
    return new_id

# --- Image Processing ---
def process_crop_image(image_path, output_path):
    """Opens a crop image, flips it vertically, rotates if needed, and saves it."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if img.height > img.width:
            img = img.rotate(-90, expand=True)
        img.save(output_path)
        print(f"Processed and saved crop image to {output_path}")

def process_whole_image(image_path, output_path):
    """Opens a whole image, rotates it -90 degrees, and saves it."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.rotate(-90, expand=True)
        img.save(output_path)
        print(f"Rotated and saved whole image to {output_path}")

def get_book_details_from_gpt(image_path):
    """Sends an image to the OpenAI API and returns the extracted book details."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai.api_key}"}
    payload = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the book title, subtitle, and author from this image of a book spine. Format as: Title: [The Title]\nSubtitle: [The Subtitle]\nAuthor: [The Author]. Write \"None\" for any missing fields."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }], "max_tokens": 300
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return parse_gpt_response(content)
    except Exception as e:
        print(f"Error calling or parsing OpenAI API: {e}")
        return "Error", "Error", "Error"

def parse_gpt_response(response_text):
    """Parses the structured response from the OpenAI API."""
    title, subtitle, author = "None", "None", "None"
    for line in response_text.strip().split('\n'):
        if line.lower().startswith("title:"):
            title = line[len("title:"):].strip()
        elif line.lower().startswith("subtitle:"):
            subtitle = line[len("subtitle:"):].strip()
        elif line.lower().startswith("author:"):
            author = line[len("author:"):].strip()
    return title, subtitle, author

create_db_if_not_exists()


@app.route('/upload_whole', methods=['POST'])
def upload_whole_image():
    """Endpoint for uploading the whole image and its bounding box data."""
    required_fields = ['file', 'whole_image_id', 'bounding_boxes_json']
    # Check that all required fields are present in the multipart request
    if not all(field in request.form or field in request.files for field in required_fields):
        error_msg = f"Request must include 'file', 'whole_image_id', and 'bounding_boxes_json'"
        return jsonify({"error": error_msg}), 400

    file = request.files['file']
    whole_image_id = request.form['whole_image_id']
    bounding_boxes_json = request.form['bounding_boxes_json']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    temp_path = os.path.join(upload_folder_whole, "temp_" + filename)
    final_path = os.path.join(upload_folder_whole, filename)
    
    file.save(temp_path)
    process_whole_image(temp_path, final_path)
    os.remove(temp_path)
    print(f"Saved rotated whole image to {final_path}")

    # Save the record, including the bounding box JSON, to the database
    db_id = insert_whole_image(whole_image_id, filename, bounding_boxes_json)
    print(f"Saved whole image record to DB with ID: {db_id} for whole_image_id: {whole_image_id}")

    return jsonify({
        "status": "success",
        "message": "Whole image and bounding boxes uploaded successfully.",
        "image_name": filename
    }), 201

@app.route('/upload', methods=['POST'])
def upload_crop_and_extract():
    """Endpoint for uploading individual, processed book spine crops."""
    if 'file' not in request.files or 'whole_image_id' not in request.form:
        return jsonify({"error": "Request must include 'file' and 'whole_image_id'"}), 400

    file = request.files['file']
    whole_image_id = request.form['whole_image_id']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    temp_path = os.path.join(upload_folder_crops, "temp_" + filename)
    processed_path = os.path.join(upload_folder_crops, filename)

    file.save(temp_path)
    process_crop_image(temp_path, processed_path)
    os.remove(temp_path)

    # print(f"Querying OpenAI API for details of crop: {filename}...")
    # title, subtitle, author = get_book_details_from_gpt(processed_path)
    title = "LATER"
    subtitle = "LATER"
    author = "LATER"

    if title == "Error":
        print(f"OpenAI API failed to extract details for {filename}.")
        return jsonify({"source": "OpenAI API (Failed)", "error": "Could not extract book details."}), 500

    # Save the crop's metadata to the database, including the linking ID
    new_id = insert_book_data(title, author, subtitle, filename, whole_image_id)
    print(f"Saved new crop entry to DB with ID: {new_id} (linked to whole_image_id: {whole_image_id})")

    return jsonify({
        "source": "OpenAI API",
        "data": {
            "id": new_id,
            "image_name": filename,
            "title": title,
            "subtitle": subtitle,
            "author": author,
            "whole_image_id": whole_image_id
        }
    })

# --- Main Execution ---
if __name__ == '__main__':
    # Runs the Flask application
    app.run(debug=True, host='0.0.0.0', port=5001)