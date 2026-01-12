import cv2
import threading
import os
import mysql.connector
from datetime import datetime
from ultralytics import YOLO
import time



model = YOLO("yolov8n.pt")

rtsp_url = "RTSP URL of Camera"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


BASE_DIR = r"D:\Internship document\Captures"
IN_DIR = os.path.join(BASE_DIR, "in")
OUT_DIR = os.path.join(BASE_DIR, "out")
os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

db = mysql.connector.connect(
    host="localhost",
    user = "root",
    password = "YOUR PASSWORD",
    database = "DB"
)

cursor = db.cursor()

def insert_db(in_path=None, out_path=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = "insert into Detection_person_entries(times, in_path, out_path) values (%s, %s, %s)"
    cursor.execute(sql, (ts, in_path, out_path))
    db.commit()

# ================= LINE CONFIG =================
LINE_Y = 250
LINE_X_Start = 600
LINE_X_End = 950
DIST_THRESHOLD = 50

# ================= COUNTER =================
count = 0
prev_centers = []

# ================= THREAD SHARED DATA =================
latest_frame = None
lock = threading.Lock()
running = True

# ================= HELPERS =================
def get_side(y):
    return "top" if y < LINE_Y else "bottom"

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def save_image(frame, direction):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if direction == "in":
        filename = f"in_{timestamp}.jpg"
        abs_path = os.path.join(IN_DIR, filename)
        rel_path = f"in/{filename}"
    else:
        filename = f"out_{timestamp}.jpg"
        abs_path = os.path.join(OUT_DIR, filename)
        rel_path = f"out/{filename}"

    cv2.imwrite(abs_path, frame)
    return rel_path

# ================= FRAME READER THREAD =================
def frame_reader():
    global latest_frame, running
    while running:
        ret = cap.grab()   
        if not ret:
            time.sleep(0.01)
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        with lock:
            latest_frame = frame

threading.Thread(target=frame_reader, daemon=True).start()


while True:
    with lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    frame = cv2.resize(frame, (1530, 800))
    results = model(frame, classes=[0], verbose=False)

    curr_centers = []

    # Draw line
    cv2.line(frame, (LINE_X_Start, LINE_Y),
             (LINE_X_End, LINE_Y), (0, 0, 255), 3)

    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            curr_centers.append((cx, cy))
            current_side = get_side(cy)

            # Draw detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            for px, py in prev_centers:
                if distance((cx, cy), (px, py)) < DIST_THRESHOLD:
                    prev_side = get_side(py)

                    if LINE_X_Start <= cx <= LINE_X_End:
                        # ENTRY
                        if prev_side == "top" and current_side == "bottom":
                            count += 1
                            rel_path = save_image(frame, "in")
                            insert_db(in_path=rel_path)
                            print("ENTRY | Count =", count)

                        # EXIT
                        elif prev_side == "bottom" and current_side == "top":
                            count -= 1
                            rel_path = save_image(frame, "out")
                            insert_db(out_path=rel_path)

                            print("EXIT | Count =", count)
                    break

    prev_centers = curr_centers.copy()

    cv2.putText(frame, f"Count: {count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("Detection-only Direction Count", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        running = False
        time.sleep(0.2) 
        break


# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()