import cv2 as cv
import pickle

img_path = './Data/test_video_screenshot.png'

# Parking lot list
try:
    with open('coordinats.pickle','rb') as f:
        pos_list = pickle.load(f)
except:
    pos_list = []

# Click and draw function
def click_it(events, x, y, flags, params):
    
    # Left click adds a rectangle box
    if events == cv.EVENT_LBUTTONDOWN:
            pos_list.append((x, y))
    # Mouse button delete last added rectangle box
    if events == cv.EVENT_MBUTTONDOWN:
            pos_list.pop()
    print(pos_list)
    # Update folder every steps
    with open('coordinats.pickle', 'wb') as f:
        pickle.dump(pos_list, f)


while True:
    img = cv.imread(img_path)

    # drawing points
    for pos in pos_list:
        cv.rectangle(img, pos, (pos[0]+50, 
                                pos[1]+100), 
                                (255,0,255), 2)    
    cv.imshow('Selected Area', img)
    cv.setMouseCallback('Selected Area', click_it)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break