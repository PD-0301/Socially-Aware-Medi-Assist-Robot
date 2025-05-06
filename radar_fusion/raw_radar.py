import serial
import struct
import time
import cv2
import numpy as np
import csv

# === Radar Serial Setup ===
RADAR_PORT = '/dev/ttyACM1'
BAUD = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# === Camera Intrinsics (K) ===
K = np.array([
    [386.59464185, 0, 385.10781334],
    [0, 379.00255236, 274.90390815],
    [0, 0, 1]
])

# === Radar to Camera Transformation Matrix (4x4) ===
T_radar_to_cam = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, -0.048],
    [0.0, 1.0,  0.0, -0.03],
    [0.0, 0.0,  0.0,  1.0]
])

# === CSV Logger Setup ===
csv_file = open("radar_camera_log.csv", mode="w", newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["cam_time", "radar_time", "x", "y", "z", "velocity", "pixel_u", "pixel_v"])

# === Video Writer Setup ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter('radar_camera_output.avi', fourcc, 20.0, (800, 600))

def parse_frame(buffer):
    idx = buffer.find(MAGIC_WORD)
    if idx == -1 or len(buffer) < idx + 48:
        return [], buffer
    try:
        pkt_len = struct.unpack('<I', buffer[idx+12:idx+16])[0]
        num_obj = struct.unpack('<H', buffer[idx+28:idx+30])[0]
        obj_start = idx + 48
        end = idx + pkt_len
        if len(buffer) < end:
            return [], buffer
        points = []
        for i in range(num_obj):
            base = obj_start + i * 16
            x, y, z, v = struct.unpack('<ffff', buffer[base:base+16])
            points.append((x, y, z, v))
        return points, buffer[end:]
    except Exception as e:
        print("[ERROR]", e)
        return [], buffer[idx+8:]

def draw_radar_points_on_image(frame, radar_points, cam_time, radar_time):
    for x, y, z, v in radar_points:
        radar_point = np.array([[x], [y], [z], [1]])  # homogeneous coords
        cam_point = T_radar_to_cam @ radar_point
        X, Y, Z = cam_point[0][0], cam_point[1][0], cam_point[2][0]
        if Z <= 0:
            continue
        u = int((K[0, 0] * X / Z) + K[0, 2])
        v_ = int((K[1, 1] * Y / Z) + K[1, 2])
        if 0 <= u < frame.shape[1] and 0 <= v_ < frame.shape[0]:
            cv2.circle(frame, (u, v_), 4, (0, 0, 255), -1)
            csv_writer.writerow([cam_time, radar_time, x, y, z, v, u, v_])
    return frame

# === Initialize Camera and Radar ===
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
ser = serial.Serial(RADAR_PORT, BAUD, timeout=0.5)
rx_buffer = b""

try:
    while True:
        # Camera frame and timestamp
        ret, frame = cap.read()
        cam_time = time.time()

        # Radar points and timestamp
        rx_buffer += ser.read(4096)
        radar_points, rx_buffer = parse_frame(rx_buffer)
        radar_time = time.time()

        if radar_points and abs(cam_time - radar_time) < 0.05:
            result = draw_radar_points_on_image(frame, radar_points, cam_time, radar_time)
            cv2.putText(result, f"Δt: {abs(cam_time - radar_time):.3f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Radar-Camera Live Sync", result)
            video_out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Interrupted by user.")
finally:
    cap.release()
    ser.close()
    video_out.release()
    csv_file.close()
    cv2.destroyAllWindows()
