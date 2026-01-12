from flask import Flask, render_template, send_from_directory
import mysql.connector
import os

app = Flask(__name__)

CAPTURES_FOLDER = r"D:\Internship document\Captures"

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="YOUR Password",
    database="DB"
)

@app.route("/")
def index():
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, times, in_path, out_path
        FROM detection_person_entries
        ORDER BY times
    """)
    records = cursor.fetchall()
    cursor.close()
    return render_template("index.html", records=records)

@app.route("/show/<path:filename>")
def serve_images(filename):
    return send_from_directory(CAPTURES_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
