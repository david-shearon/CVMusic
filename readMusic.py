import cv2
import numpy as np
import sys
from playMusic import play_music

#testing music functionality
play_music([['C4', 2], ['E4', 4], ['F5', 1], ['A3', 2]], "test", 120)

mouse_list = []
x = 0
y = 0
myFlag = 0

# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))
        
        #only 4 corners
        marker_points = param
        if(len(marker_points) > 4):
            marker_points = marker_points[-4:]
        
        print("Len(param)", len(param))
        marker_img = np.copy(input_img)
        for point in marker_points:
            cv2.drawMarker(marker_img, (point[0], point[1]), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.imshow(window_name, marker_img)

#input image and scale for usability
MAX_WIDTH = 800
input_img = cv2.imread("./images/1.jpg")
if input_img.shape[1] > MAX_WIDTH:
        s = MAX_WIDTH / input_img.shape[1]
        input_img = cv2.resize(input_img, dsize=None, fx=s, fy=s)

#Allow user to find corners (temporary fix)
window_name = "Chose Corners"
cv2.namedWindow(window_name)
cv2.imshow(window_name, input_img)
cv2.setMouseCallback(window_name=window_name, on_mouse=get_xy, param=mouse_list)
cv2.waitKey(0)
print("List of corners", mouse_list)

#To scale up/donw the sheet size
sheet_size_multiplier = 100

H,_ = cv2.findHomography(np.array(mouse_list), np.array([(0, 0), (8.5 * sheet_size_multiplier, 0), (8.5 * sheet_size_multiplier, 11 * sheet_size_multiplier), (0, 11 * sheet_size_multiplier)]))
warped = cv2.warpPerspective(input_img, H, (int(8.5 * sheet_size_multiplier), int(11 * sheet_size_multiplier)))

cv2.imshow("Output Image", warped)
cv2.waitKey(0)