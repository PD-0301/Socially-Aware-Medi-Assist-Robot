import cv2
import os

cap = cv2.VideoCapture(2)  # Use your Arducam index
cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
save_dir = "calib_images"
os.makedirs(save_dir, exist_ok=True)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):  # Press 's' to save
        filename = os.path.join(save_dir, f"calib_{count:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
