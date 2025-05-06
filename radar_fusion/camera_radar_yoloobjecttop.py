import serial, struct, math, numpy as np, cv2, time
from ultralytics import YOLO

# === CONFIGURATION ===
RADAR_PORT = '/dev/ttyACM1'
BAUD = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# === Camera Intrinsics ===
K = np.array([[386.59464185, 0., 385.10781334],
              [0., 379.00255236, 274.90390815],
              [0., 0., 1.]])
D = np.array([-0.17272455, 0.05733935, -0.00293362, -0.00081243, -0.04610971])

T_radar_to_cam = np.array([[1., 0.,  0.,   0.],
                           [0., 0., -1., -0.048],
                           [0., 1.,  0., -0.03],
                           [0., 0.,  0.,  1.]])

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

radar_ser = serial.Serial(RADAR_PORT, BAUD, timeout=0.01)
model = YOLO('yolov8n.pt')

def get_radar_points():
    data = radar_ser.read(4096)
    idx = data.find(MAGIC_WORD)
    points = []
    if idx != -1 and len(data) > idx + 48:
        try:
            pkt_len = struct.unpack('<I', data[idx+12:idx+16])[0]
            num_obj = struct.unpack('<H', data[idx+28:idx+30])[0]
            obj_start = idx + 48
            for i in range(num_obj):
                base = obj_start + i * 16
                if len(data) >= base + 16:
                    x, y, z, v = struct.unpack('<ffff', data[base:base+16])
                    if 0.1 < math.sqrt(x**2 + y**2 + z**2) < 3.0:
                        points.append([x, y, z])
        except Exception as e:
            print(f"[Radar Parse Error] {e}")
    return np.array(points)

def pixel_to_world(x, y, depth):
    inv_K = np.linalg.inv(K)
    pixel = np.array([x, y, 1.0])
    cam_coords = inv_K @ pixel * depth
    return cam_coords

def radar_in_bbox(radar_pixels, bbox):
    x1, y1, x2, y2 = bbox
    count = 0
    for x, y in radar_pixels:
        if x1 <= x <= x2 and y1 <= y <= y2:
            count += 1
    return count >= 2

smoothing_factor = 0.8
prev_radius = 0

try:
    while True:
        ret, frame = cap.read()
        radar_points = get_radar_points()
        if not ret:
            continue

        radar_pixels = []
        cam_points = []
        if radar_points.shape[0] > 0:
            radar_hom = np.hstack((radar_points, np.ones((radar_points.shape[0], 1))))
            cam_points = (T_radar_to_cam @ radar_hom.T).T[:, :3]
            cam_points = cam_points[cam_points[:, 2] > 0]
            if cam_points.shape[0] > 0:
                obj_pts = cam_points.reshape(-1, 1, 3).astype(np.float32)
                img_pts, _ = cv2.projectPoints(obj_pts, np.zeros((3,1)), np.zeros((3,1)), K, D)
                radar_pixels = np.array(img_pts.reshape(-1, 2))

        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
        top_view = np.ones((500, 500, 3), dtype=np.uint8) * 30

        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = results.names[int(cls)]

            if len(radar_pixels) > 0 and radar_in_bbox(radar_pixels, (x1, y1, x2, y2)):
                matched = []
                for (px, py), (x, y, z) in zip(radar_pixels, cam_points):
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        matched.append([x, z])
                if matched:
                    matched = np.array(matched)
                    cx, cz = matched.mean(axis=0) * 100  # cm
                    depth = np.linalg.norm(matched.mean(axis=0))

                    # --- World size estimation from corners ---
                    tl = pixel_to_world(x1, y1, depth)
                    br = pixel_to_world(x2, y2, depth)
                    real_w = abs(br[0] - tl[0])
                    real_h = abs(br[1] - tl[1])
                    radius = int(np.sqrt(real_w**2 + real_h**2) * 50)  # scaling for visual

                    # Smooth radius
                    radius = int(smoothing_factor * prev_radius + (1 - smoothing_factor) * radius)
                    prev_radius = radius

                    cv2.circle(top_view, (int(250+cx), int(250-cz)), radius, (0, 255, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.putText(frame, f"{label}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        for (u, v) in np.array(radar_pixels).astype(int):
            if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                cv2.circle(frame, (u, v), 3, (0, 255, 0), -1)

        cv2.imshow("Radar-Camera Fusion", frame)
        cv2.imshow("Top-Down View", top_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted.")
finally:
    cap.release()
    radar_ser.close()
    cv2.destroyAllWindows()
