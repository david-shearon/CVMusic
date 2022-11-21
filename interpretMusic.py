import cv2
import numpy as np

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

print(len(houghLines))

# NOTE: need a way to merge similar Hough lines, as the edge detector will output two on average for every staff line

# print the detected lines from the hough transform
lines_image = raw_image.copy()
for line in houghLines:
    cv2.line(lines_image, line[0], line[1], (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)


cv2.imwrite("./test_images/detected_lines.jpg", lines_image)
cv2.waitKey(0)