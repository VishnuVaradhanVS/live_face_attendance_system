import face_recognition as fr
import os
import pickle
import sqlite3

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pickle"

knownEncodings = []
names = []

conn = sqlite3.connect("database.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT)")
cursor.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, name TEXT, timestamp TEXT)")

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        print(f"Processing {name}'s image...")

        image = fr.load_image_file(image_path)
        face_locations = fr.face_locations(image)

        if face_locations:
            encoding = fr.face_encodings(image)[0]
            knownEncodings.append(encoding)
            names.append(name)

            cursor.execute("INSERT OR IGNORE INTO students (name) VALUES (?)", (name,))
        else:
            print(f"⚠ No face detected in {filename}, skipping...")

conn.commit()
conn.close()

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": knownEncodings, "names": names}, f)

print(f"✅ Encodings saved to {ENCODINGS_FILE}")
