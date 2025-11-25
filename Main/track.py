import os    
os.environ["QT_QPA_PLATFORM"] = "xcb"  # if you are using Ubuntu 22.04
import cv2 as cv
import pickle
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "..", "Data", "test_vid_short.mp4")
file_path = 'coordinates.pickle'
print(video_path)
# Parking lot list
try:
    with open(file_path, 'rb') as f:
        pos_list = pickle.load(f)
except:
    pos_list = []

# Temp list to keep track of clicks
current_points = []

# Click and draw function
def click_it(events, x, y, flags, params):
    global current_points, pos_list
    
    # Left click adds a point
    if events == cv.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

            if len(current_points) == 4: # 4 point check
                 pos_list.append(current_points)
                 current_points = [] # Reset the list      

                 # Save
                 with open(file_path, 'wb') as f:
                      pickle.dump(pos_list, f)  
    
    # Mouse button delete last added rectangle box
    if events == cv.EVENT_MBUTTONDOWN:
            if len(pos_list) > 0 : 
                pos_list.pop()
                
                # Save again
                with open(file_path, 'wb') as f:
                     pickle.dump(pos_list, f)
            current_points = [] # Reset the list for half lines

cap = cv.VideoCapture(video_path)
success, base_img = cap.read() # Read only one frame
cap.release()

if not success:
     print("Error: Check the video path")

cv.namedWindow('Selected Area', cv.WINDOW_NORMAL)
cv.resizeWindow('Selected Area', 1200, 720)
cv.setMouseCallback('Selected Area', click_it)

while True:
    img = base_img.copy()

    # Drawing points
    for pos in pos_list:
        # For Polylines : numpy array
        pts = np.array(pos, np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv.polylines(img, [pts], True, (0, 255, 0), 2)
    # Shows while drawing points
    if len(current_points) > 0:
         pts_temp = np.array(current_points, np.int32)
         pts_temp = pts_temp.reshape((-1, 1, 2))

         cv.polylines(img, [pts_temp], False, (0, 0 , 255), 2) # isClosed=False

         # Display edge points       
         for pt in current_points:
            cv.circle(img, pt, 3, (0,0,255), -1)

    cv.imshow('Selected Area', img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    
cv.destroyAllWindows()