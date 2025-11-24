import cv2 as cv
import pickle
import numpy as np

img_path = './Data/test_video_screenshot.png'
file_path = 'coordinates.pickle'
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

            if len(current_points) == 4: # 4 pount check
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


while True:
    img = cv.imread(img_path)

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
    cv.setMouseCallback('Selected Area', click_it)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break