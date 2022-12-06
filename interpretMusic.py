import cv2
import numpy as np
import os

def getNoteLetter(top_line, bottom_line, x, y):
    notes_list = ['E', 'F', 'G', 'A', 'B', 'C', 'D', 'E', 'F']

    up_left = top_line[0]
    up_right = top_line[1]

    down_left = bottom_line[0]
    down_right = bottom_line[1]

    slope = float(up_left[1] - up_right[1]) / float(up_left[0] - up_right[0])

    upper_y = up_left[1] - slope * (x - up_left[0])
    lower_y = down_left[1] - slope * (x - down_left[0])

    note_index = int(np.round(float(y - lower_y) * 9 / float(upper_y - lower_y)) - 1)
    return notes_list[note_index]


SIGMA_BLUR = 1.0
# adjustable parameters to fine tune the edge detector
MIN_EDGES_FRACTION = 0.04
MAX_EDGES_FRACTION = 0.05
# kind of a throwaway parameter, with the below, the only lines detected are the staff lines
MIN_HOUGH_VOTES_FRACTION = 0.10
# only look for really long continuous lines (really reliably picks out the staff lines and nothing else)
MIN_LINE_LENGTH_FRACTION = 0.75
# adjustable parameter to allow lines to be detected with discontinuities (up to this times the image width)
MAX_LINE_GAP_FRACTION = 0.05

# create a directory for the output images in this file if one doesn't exist
if not os.path.exists("./test_images/"):
    os.mkdir("./test_images/")

# read in the raw image with the page of sheet music
raw_image = cv2.imread("./images/5.jpg")
annotated_image = raw_image.copy()
raw_image_height, raw_image_width, _ = raw_image.shape

# convert image to grayscale
gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

# Smooth the image with a Gaussian filter.  If sigma is not provided, it
# computes it automatically using   sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.
# NOTE: blur only really necessary on real world images, for digitally scanned music, leaving this out should be fine
# gray_image = cv2.GaussianBlur(
#     src=gray_image,
#     ksize=(0, 0),       # kernel size (should be odd numbers; if 0, compute it from sigma)
#     sigmaX=SIGMA_BLUR, 
#     sigmaY=SIGMA_BLUR
# )

# threshold the image to isolate the page of music
# - assuming a white background of the music, it should stand out clearly from the background
_, thresh_image = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)

# create a simple kernel that is a rectangle 0.1% the width of the image
kernelSize = int(max((2, 0.001*thresh_image.shape[1])))
kernelShape = (kernelSize,kernelSize)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelShape)

# # perform a morphological operations on the image to clean up small noise
# thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

# use the thresholded image to throw out the background of the raw image for better line detection
gray_image[thresh_image == 0] = 0

cv2.imwrite("./test_images/thresholded_image.jpg", thresh_image)
cv2.imwrite("./test_images/annotated_image.jpg", gray_image)
cv2.waitKey(0)

# run the canny edge detector, adjusting the threshold until the number of edges
# detected falls within the specified edge density
thresh_canny = 1.0

edge_image = cv2.Canny(
    image=gray_image,
    apertureSize=3,  # size of Sobel operator
    threshold1=thresh_canny,  # lower threshold
    threshold2=3*thresh_canny,  # upper threshold
    L2gradient=True  # use more accurate L2 norm
)

while np.sum(edge_image)/255 < MIN_EDGES_FRACTION * (raw_image_width * raw_image_height):
    thresh_canny *= 0.9
    edge_image = cv2.Canny(
        image=gray_image,
        apertureSize=3,  # size of Sobel operator
        threshold1=thresh_canny,  # lower threshold
        threshold2=3*thresh_canny,  # upper threshold
        L2gradient=True  # use more accurate L2 norm
    )
while np.sum(edge_image)/255 > MAX_EDGES_FRACTION * (raw_image_width * raw_image_height):
    thresh_canny *= 1.1
    edge_image = cv2.Canny(
        image=gray_image,
        apertureSize=3,  # size of Sobel operator
        threshold1=thresh_canny,  # lower threshold
        threshold2=3*thresh_canny,  # upper threshold
        L2gradient=True  # use more accurate L2 norm
    )

#cv2.imshow("detected edges", edge_image)
cv2.imwrite("./test_images/detected_edges.jpg", edge_image)
cv2.waitKey(0)

# output shape is N, 2, 2 where N is the number of lines detected, each one having two endpoints with their own dimension (simplifies line printing later)
houghLines = cv2.HoughLinesP(
    image=edge_image,
    rho=1,
    theta=np.pi/6, # reduce the number of angles to search through since lines will be nominally horizontal
    threshold=0, #int(edge_image.shape[1] * MIN_HOUGH_VOTES_FRACTION), NOTE: could reintroduce this, but given other parameters, the only lines detected are the ones we want
    lines=None,
    minLineLength=int(edge_image.shape[1] * MIN_LINE_LENGTH_FRACTION),
    maxLineGap=int(edge_image.shape[1] * MAX_LINE_GAP_FRACTION)
).squeeze().reshape(-1, 2, 2) # squeeze to remove unneccesary extra dimensions in the list, then reshape to put each endpoint in it's own array

def euclid_distance(line_one, line_two):
    return np.sqrt(np.sum((line_one - line_two) ** 2, axis=0))

# Merges all lines that are within a range of each other
SIMILARITY_THRESH = 3
cleaned_lines = np.empty((0,2,2), np.int32)
while 0 < len(houghLines):
    # Finds all lines that are similar to the current line and removes them from the list
    same_lines = np.empty((0,2,2))
    curr_line = houghLines[0]
    houghLines = np.delete(houghLines, 0, axis=0)
    for i, line in enumerate(houghLines):
        # Test if the euclidian distance between corresponding endpoints is less than the threshold and add to same_lines if so
        if (euclid_distance(curr_line, line) < SIMILARITY_THRESH).all():
            houghLines = np.delete(houghLines, i - same_lines.shape[0], axis=0)
            same_lines = np.vstack((same_lines, line.reshape((-1, 2, 2))))
    
    # Calculate the average of the similar lines and add to cleaned_lines
    new_line = np.int32((curr_line + np.sum(same_lines, axis=0)) / (same_lines.shape[0] + 1))
    cleaned_lines = np.vstack((cleaned_lines, new_line.reshape((-1, 2, 2))))

# Group the lines into staff bounding boxes
line_groups = np.empty((0, 5), dtype=np.int32)
staff_boxes = np.empty((0, 4, 2), dtype=np.int32)
for curr_line in cleaned_lines:
    # Create an array of tuples with the index, euclidean distance from the current line, and the y position of every line
    dists = np.array([
        (i, np.mean(euclid_distance(curr_line, line)), np.mean(line[:, 1])) for i, line in enumerate(cleaned_lines)
        ],
        dtype=[("i", int), ("dist", float), ("y", float)])
    # Get the indexes of the 5 smallest euclidean distance lines, sorted by their y position
    staff = np.array([i for i, _, _ in np.sort(np.sort(dists, order="dist")[:5], order="y")])

    # If it is a new staff, store and get the bounding box
    if staff not in line_groups:
        line_groups = np.vstack((line_groups, staff))
        staff_boxes = np.vstack((staff_boxes, cleaned_lines[staff[::4]].reshape(-1, 4, 2)))

# print the detected lines from the hough transform
lines_image = raw_image.copy()

# prepare binary image for quarter note detection
# thresh, binary_img = cv2.threshold(raw_image.copy(), thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, close_image = cv2.threshold(raw_image.copy(), thresh=150, maxval=255, type=cv2.THRESH_BINARY)
close_image = cv2.bitwise_not(close_image)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
close_image = cv2.morphologyEx(close_image, cv2.MORPH_OPEN, kernel)
# close_image = cv2.morphologyEx(close_image, cv2.MORPH_CLOSE, kernel)

# Find quarter note locations
cv2.imshow("boxes", close_image)
cv2.waitKey(0)
num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(close_image))
for n in range(num_labels):
    xc, yc = centroids[n]
    area = stats[n, cv2.CC_STAT_AREA]
    print("Region %d, centroid: (%f,%f), area = %d" % (n, xc, yc, area))

bgr_image_display = cv2.cvtColor(close_image, cv2.COLOR_GRAY2BGR)
for stat, centroid in zip(stats, centroids):
    x0 = stat[cv2.CC_STAT_LEFT]
    y0 = stat[cv2.CC_STAT_TOP]
    w = stat[cv2.CC_STAT_WIDTH]
    h = stat[cv2.CC_STAT_HEIGHT]
    bgr_image_display = cv2.rectangle(
        img=bgr_image_display, pt1=(x0, y0), pt2=(x0 + w, y0 + h),
        color=(0, 0, 255), thickness=1)
cv2.imshow("boxes", bgr_image_display)
cv2.waitKey(0)




# sort the line array from top to bottom
def myFunc(e):
    return float(e[0][1] + e[1][1])/2
cleaned_lines_list = list(cleaned_lines)
cleaned_lines_list = sorted(cleaned_lines_list, key=myFunc)

# Draw top/bottom lines
for line_count in range(len(cleaned_lines_list)):
    staff_line = int(line_count / 5)
    if(line_count % 5 == 0):
        cv2.line(lines_image, cleaned_lines_list[line_count][0], cleaned_lines_list[line_count][1], (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(lines_image, cleaned_lines_list[line_count + 4][0], cleaned_lines_list[line_count + 4][1], (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.line(close_image, cleaned_lines_list[line_count][0], cleaned_lines_list[line_count][1], (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(close_image, cleaned_lines_list[line_count + 4][0], cleaned_lines_list[line_count + 4][1], (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        if(staff_line == 1):
            print(getNoteLetter(cleaned_lines_list[line_count], cleaned_lines_list[line_count + 4], 223, 428))
        # cleaned_lines_list.remove(line)



cv2.imwrite("./test_images/detected_lines.jpg", lines_image)
cv2.imwrite("./test_images/lines_and_dots.jpg", close_image)
cv2.waitKey(0)

def scale_template_images(top_line, bottom_line, template_images):
    # Input: the y position of the top line of any scale, the y position of the bottom line of the same scale,
    # and an array of note images to be used for template matching
    # Output: array of scaled template images to be actually used in template matching, ordered the same as before
    scale_height = top_line - bottom_line
    # we know from the size of the scale how big a note should be approximately - 1/4 of the scale's height
    note_height = np.ceil(scale_height / 4)
    scaled_images = []
    for image in template_images:
        width_height_ratio = len(image[0]) / len(image)
        new_height = note_height * width_height_ratio
        scaled_image = cv2.resize(image, (new_height, note_height), interpolation=cv2.INTER_LINEAR)
        scaled_images.append(scaled_image)

    return scaled_images