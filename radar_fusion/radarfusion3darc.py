import serial
import struct
import math
import numpy as np
import cv2
import time
from ultralytics import YOLO

# === Radar Config ===
PORT = '/dev/ttyACM1'
BAUD = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
DIST_THRESHOLD = 1.0  # meters
VEL_THRESHOLD = 2.0   # m/s

# === Camera Config ===
CAM_INDEX = 2
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for better accuracy

# === Serial Connection ===
radar = serial.Serial(PORT, BAUD, timeout=0.5)

# === Radar Parser ===
def get_radar_points():
    data = radar.read(8192)
    idx = data.find(MAGIC_WORD)
    points = []

    if idx != -1 and len(data) > idx + 48:
        try:
            pkt_len = struct.unpack('<I', data[idx+12:idx+16])[0]
            num_obj = struct.unpack('<H', data[idx+28:idx+30])[0]
            obj_start = idx + 48
            for i in range(num_obj):
                start = obj_start + i * 16
                if len(data) >= start + 16:
                    x, y, z, v = struct.unpack('<ffff', data[start:start+16])
                    dist = math.hypot(x, y)
                    if dist < DIST_THRESHOLD and abs(v) > VEL_THRESHOLD:
                        points.append((x, y, v))
        except:
            pass
    return points

# === YOLO Inference ===
def detect_camera_objects(frame):
    results = model(frame, verbose=False)[0]
    classes = [model.names[int(c)] for c in results.boxes.cls]
    return classes

# === Main Loop ===
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        radar_points = get_radar_points()
        ret, frame = cap.read()
        if not ret:
            continue

        # Object detection
        detected_objects = detect_camera_objects(frame)

        # === Decision Fusion ===
        if radar_points:
            print("📡 Radar: Detected object(s) within 1m")

            if 'person' in detected_objects:
                print("🛑 Human Detected → STOP")
                action = "STOP"
            elif any(cls in detected_objects for cls in ['chair', 'bench', 'tv']):
                print("↩️ Furniture Detected → TURN")
                action = "TURN"
            else:
                print("🐢 Unknown Obstacle → SLOW DOWN")
                action = "SLOW"
        else:
            print("✅ Clear Path → MOVE FORWARD")
            action = "MOVE"

        # Show action on frame
        cv2.putText(frame, f"Action: {action}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Fusion View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Shutting down...")

finally:
    cap.release()
    radar.close()
    cv2.destroyAllWindows()
