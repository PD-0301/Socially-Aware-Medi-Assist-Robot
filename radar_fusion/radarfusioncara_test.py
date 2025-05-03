import serial
import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import time

# === CONFIG ===
RADAR_PORT = '/dev/ttyACM1'
ESP32_PORT = '/dev/ttyUSB0'
RADAR_BAUD = 921600
ESP32_BAUD = 115200

# === SERIAL SETUP ===
esp32 = serial.Serial(ESP32_PORT, ESP32_BAUD)
radar = serial.Serial(RADAR_PORT, RADAR_BAUD, timeout=0.5)
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# === CAMERA + YOLO SETUP ===
cap = cv2.VideoCapture(2)
model = YOLO("yolov8n.pt")  # Use yolov8n for speed

# === DUMMY RADAR PARSER ===
def parse_radar_data(data):
    # Simulated dummy data — replace with actual parser
    num_points = np.random.randint(10, 20)
    return np.random.uniform(-1, 1, (num_points, 4))  # (x, y, z, v)

# === KALMAN FILTER SETUP ===
def create_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])
    kf.F = np.array([[1,0,1,0],
                     [0,1,0,1],
                     [0,0,1,0],
                     [0,0,0,1]])
    kf.H = np.array([[1,0,0,0],
                     [0,1,0,0]])
    kf.P *= 1000.
    kf.R *= 0.01
    kf.Q *= 0.01
    return kf

kf = create_kalman()

# === CONTROL FUNCTION ===
def send_command(drive, steer, brake):
    try:
        cmd = f"{drive},{steer},{int(brake)}\n"
        esp32.write(cmd.encode())
    except Exception as e:
        print("ESP32 error:", e)

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # === RADAR DATA ===
    if radar.in_waiting:
        raw_data = radar.read(radar.in_waiting)
        if MAGIC_WORD in raw_data:
            radar_points = parse_radar_data(raw_data)
        else:
            radar_points = []
    else:
        radar_points = []

    # === YOLO DETECTION ===
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append((x1, y1, x2, y2, cls, conf))

    brake = False
    for point in radar_points:
        x, y, z, v = point
        distance = np.sqrt(x**2 + y**2 + z**2)

        # Simulated projection
        px = int(320 + x * 100)
        py = int(240 - y * 100)

        for (x1, y1, x2, y2, cls, conf) in detections:
            if x1 < px < x2 and y1 < py < y2:
                if distance < 0.3:
                    brake = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{distance:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # === SMOOTHING ===
    if len(radar_points) > 0:
        avg_x, avg_y = np.mean(radar_points[:,0]), np.mean(radar_points[:,1])
        kf.predict()
        kf.update([avg_x, avg_y])
        smoothed = kf.x[:2]
        cv2.circle(frame, (int(320 + smoothed[0]*100), int(240 - smoothed[1]*100)), 6, (255,0,0), -1)

    # === SEND CONTROL ===
    if brake:
        send_command(0, 0, True)
    else:
        send_command(1, 0, False)

    # === PLOT RADAR POINTS ===
    for pt in radar_points:
        x, y, z, v = pt
        px = int(320 + x * 100)
        py = int(240 - y * 100)
        cv2.circle(frame, (px, py), 4, (0,0,255), -1)

    cv2.imshow("Radar-Camera Fusion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
esp32.close()
radar.close()
