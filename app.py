from flask import Flask, render_template, Response, jsonify, request
import cv2
import face_recognition as fr
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

ENCODINGS_FILE = "encodings.pickle"

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
        knownEncodings = data.get("encodings", [])
        names = data.get("names", [])
else:
    knownEncodings = []
    names = []
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": knownEncodings, "names": names}, f)


cam = cv2.VideoCapture(0)

def clear_attendance():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, name TEXT, timestamp TEXT)")
    cursor.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()

clear_attendance()

def recognize_faces():
    while True:
        _, frame = cam.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        facelocations = fr.face_locations(rgb_frame)
        encodings = fr.face_encodings(rgb_frame, facelocations)

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        for face_location, encoding in zip(facelocations, encodings):
            top, right, bottom, left = face_location
            name = "Unknown"

            if knownEncodings:
                distances = fr.face_distance(knownEncodings, encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.6:
                    name = names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            if name != "Unknown":
                cursor.execute("SELECT * FROM attendance WHERE name = ? AND timestamp >= date('now')", (name,))
                if not cursor.fetchone():
                    cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, datetime.now()))
                    conn.commit()

        conn.close()

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance AS a WHERE id = (SELECT MIN(id) FROM attendance WHERE name = a.name) ORDER BY timestamp ASC;")
    data = cursor.fetchall()
    conn.close()
    return render_template('index.html', records=data)

@app.route('/video_feed')
def video_feed():
    return Response(recognize_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_attendance')
def get_attendance():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance AS a WHERE id = (SELECT MIN(id) FROM attendance WHERE name = a.name) ORDER BY timestamp DESC;")
    data = cursor.fetchall()
    conn.close()
    attendance_list = [{"id": row[0], "name": row[1], "timestamp": row[2]} for row in data]
    return jsonify(attendance_list)

@app.route('/attendance')
def attendance():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance AS a WHERE id = (SELECT MIN(id) FROM attendance WHERE name = a.name) ORDER BY name ASC;")
    data = cursor.fetchall()
    conn.close()
    return render_template('attendance.html', records=data)

@app.route('/new_face')
def new_face():
    return render_template('addface.html', records=data)

@app.route('/add_face', methods=['POST'])
def add_face():
    """ Capture a face, encode it, and save it to the database """
    global knownEncodings, names

    name = request.form.get("name")

    if not name:
        return jsonify({"status": "error", "message": "Name is required!"})

    _, frame = cam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    facelocations = fr.face_locations(rgb_frame)
    encodings = fr.face_encodings(rgb_frame, facelocations)

    if encodings:
        knownEncodings.append(encodings[0])
        names.append(name)

        data = {"encodings": knownEncodings, "names": names}
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)

        return jsonify({"status": "success", "message": f"Face added for {name}"})
    
    return jsonify({"status": "error", "message": "No face detected!"})

@app.route('/download_attendance')
def download_attendance():
    """Generates and downloads attendance records as a CSV file."""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name,timestamp FROM attendance AS a WHERE id = (SELECT MIN(id) FROM attendance WHERE name = a.name) ORDER BY name ASC;")
    attendance_records = cursor.fetchall()
    conn.close()

    csv_data = "Name,Timestamp\n" + "\n".join([f"{name},{timestamp}" for name, timestamp in attendance_records])

    response = Response(csv_data, mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=attendance_report.csv"
    return response

if __name__ == "__main__":
    app.run(debug=True)
