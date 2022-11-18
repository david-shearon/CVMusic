import math

import cv2
import numpy as np
import sys
from playMusic import save_music

#testing music functionality
save_music([['C4', 2], ['E4', 4], ['F5', 1], ['A3', 2]], "test", 120)

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
notes_image = cv2.warpPerspective(input_img, H, (int(8.5 * sheet_size_multiplier), int(11 * sheet_size_multiplier)))
notes_image_raw = notes_image
notes_image_original = notes_image
cv2.imshow("Output Image", notes_image)
cv2.waitKey(0)


# edge detection from lab 10
MAX_WIDTH = 600
SIGMA_BLUR = 1.0
MIN_HOUGH_VOTES_FRACTION = np.array((0.1, 0.1, 0.1))
MIN_LINE_LENGTH_FRACTION = np.array((0.1, 0.04, 0.1))
MIN_FRACT_EDGES = 0.2
MAX_FRACT_EDGES = 0.1
count = 0
thresh_canny = 20
if notes_image.shape[1] > MAX_WIDTH:
    s = MAX_WIDTH / notes_image.shape[1]
    notes_image_raw = cv2.resize(notes_image, dsize=None, fx=s, fy=s)
    notes_image = cv2.resize(notes_image, dsize=None, fx=s, fy=s)


notes_image_raw = cv2.GaussianBlur(src=notes_image, ksize=(0, 0), sigmaX=SIGMA_BLUR, sigmaY=SIGMA_BLUR)
notes_image = cv2.GaussianBlur(src=notes_image, ksize=(0, 0), sigmaX=SIGMA_BLUR, sigmaY=SIGMA_BLUR)


# Pick a threshold such that we get a relatively small number of edge points.
while np.sum(notes_image) / 255 < MIN_FRACT_EDGES * (notes_image_raw.shape[1] * notes_image_raw.shape[0]):
    print("Decreasing threshold ...")
    thresh_canny *= 0.9
    notes_image = cv2.Canny(
        image=notes_image_raw,
        apertureSize=3,  # size of Sobel operator
        threshold1=thresh_canny,  # lower threshold
        threshold2=3 * thresh_canny,  # upper threshold
        L2gradient=True)  # use more accurate L2 norm
while np.sum(notes_image) / 255 > MAX_FRACT_EDGES * (notes_image_raw.shape[1] * notes_image_raw.shape[0]):
    print("Increasing threshold ...")
    thresh_canny *= 1.1
    notes_image = cv2.Canny(
        image=notes_image_raw,
        apertureSize=3,  # size of Sobel operator
        threshold1=thresh_canny,  # lower threshold
        threshold2=3 * thresh_canny,  # upper threshold
        L2gradient=True)  # use more accurate L2 norm

cv2.imshow("Edge", notes_image)
cv2.waitKey(0)
cv2.imshow("Edge", notes_image_raw)
cv2.waitKey(0)

houghThresh = int(notes_image.shape[1] * MIN_HOUGH_VOTES_FRACTION[count])
print("houghThresh: ", houghThresh)

hough_lines = cv2.HoughLinesP(
    image=notes_image,
    rho=1,  # Distance resolution of the accumulator in pixels
    theta=math.pi / 180,  # Angle resolution of the accumulator in radians
    threshold=houghThresh,  # Accumulator threshold (get lines where votes>threshold)
    lines=None,
    minLineLength=int(notes_image.shape[1] * MIN_LINE_LENGTH_FRACTION[count]),
    maxLineGap=10
)

NLINES = 25
for i in range(min(NLINES, len(hough_lines))):
    rho = hough_lines[i][0][0]  # distance from (0,0)
    theta = hough_lines[i][0][1]  # angle in radians
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho  # Point on line, where rho vector intersects
    y0 = b * rho
    # Find two points on the line, very far away.
    p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    cv2.line(img=notes_image_original, pt1=p1, pt2=p2, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

#test = vanishing.find_vanishing_point_directions(hough_lines, notes_image_raw, num_to_find=3, K=None)
#print(test)

cv2.imwrite("edge" + str(count) + ".jpg", notes_image)
cv2.imwrite("final" + str(count) + ".jpg", notes_image_original)
count += 1
