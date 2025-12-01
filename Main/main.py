import os
import pickle
import cv2 as cv
import numpy as np
from ultralytics import YOLO

from utils import adjust_gamma, apply_clahe, increase_contrast

DEBUG_MODE = False
CONFIDENCE_THRESHOLD = 0.10

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "..", "Data", "test_video.mp4")

cap = cv.VideoCapture(video_path)

model = YOLO('yolov8n.pt') # NANO model

# Load park coordinates
try:
    with open('coordinates.pickle', 'rb') as f:
        pos_list = pickle.load(f)
except:
    print("Pickle file not found")
    pos_list = []

cv.namedWindow('Smart Parking System', cv.WINDOW_NORMAL)
cv.resizeWindow('Smart Parking System', 1200, 720)

while True:
    # Video loop
    if cap.get(cv.CAP_PROP_POS_FRAMES) == cap.get(cv.CAP_PROP_FRAME_COUNT):
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    
    ret, frame = cap.read()
    if not ret:
        break
    
    
    #gamma = adjust_gamma(frame, gamma=2.6)
    #new_frame = apply_clahe(frame, clip_limit=2.0)   
    contrast_frame = increase_contrast(frame, alpha=1.6, beta=-40)
    results = model(contrast_frame, stream=True, verbose=False, imgsz=1248)

    temp_list = [] # to hold the center points of the vehicles

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Confidence control
            conf = box.conf[0] # Tensor format data
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]: # COCO IDs

               # Get cords -- Float to int
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculate center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                temp_list.append(((cx, cy)))
                if DEBUG_MODE:
                    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0 ,255), 1)
                    cv.circle(frame, (cx, cy), 5, (0, 255 ,255), -1)
    
    empty_spaces = 0

    for pos in pos_list:
        pts = np.array(pos, np.int32).reshape((-1, 1, 2))

        is_occupied = False # Assumption that the area is empty

        for center in temp_list:

            result = cv.pointPolygonTest(pts, center, False)

            if result >= 0: # If any car is in the space
                is_occupied = True
                break

        if is_occupied:
            color = (0, 0, 255) # Red: If the space is occupied
            thickness = 2
        else:
            color = (0, 255, 0) # Green : If the space is empyt
            thickness = 2
            empty_spaces += 1
        
        cv.polylines(frame, [pts], True, color, thickness)

    cv.imshow("Smart Parking System", frame)
    


    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        DEBUG_MODE = not DEBUG_MODE

cap.release()
cv.destroyAllWindows()

