import pickle
import cv2 as cv
import numpy as np
from ultralytics import YOLO

DEBUG_MODE = False
cap = cv.VideoCapture('./Data/test_video.mp4')
model = YOLO('yolov8n.pt') # Nano version for coco dataset

# Load park coordinates
with open('coordinates.pickle', 'rb') as f:
    pos_list = pickle.load(f)

def check_park(img, prop_img):

    for pos in pos_list:
        # Numpy array transform
        pts = np.array(pos, np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv.polylines(img, [pts], True, color=(0, 255 ,0), thickness=2)

    return img

while True:
    # Video loop
    if cap.get(cv.CAP_PROP_POS_FRAMES) == cap.get(cv.CAP_PROP_FRAME_COUNT):
        cap.set(cv.CAP_PROP_POS_FRAMES)
    
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    temp_list = [] # to hold the center points of the vehicles

    for r in results:
        boxes = r.boxes
        for box in boxes:

            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]: # COCO IDs

                # Get cords
                x1, y1, x2, y2 = box.xyxy[0]

                # Calculate center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                temp_list.append(((cx, cy)))
                if DEBUG_MODE:
                    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0 ,255), 1)
                    cv.circle(frame, (cx, cy), 5, (0, 255 ,255), -1)
    
    for pos in pos_list:
        pts = np.array(pos, np.int32)
        pts = pts.reshape((-1, 1, 2))

        for center in temp_list:

            result = cv.pointPolygonTest(pts, center, False)

        cv.polylines(frame, [pts], True, (0, 0, 255), 2 )
    
    cv.imshow("Smart Parking System", frame)

    key = cv.waitKey(60) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        DEBUG_MODE = not DEBUG_MODE

cap.release()
cv.destroyAllWindows()

