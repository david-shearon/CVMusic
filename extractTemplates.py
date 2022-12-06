import cv2
import numpy as np
import interpretMusic

# NOTE: BIG caveat with this approach is that currently it only robustly works with image 5,
#       for the scope of this project I say we just get it working with image 5 and worry about
#       adding the ability to detect in other images as a bonus if we get there, too much of what
#       we're doing from the matching to the line detection depends on manually tuned parameters

raw_image = cv2.imread("./images/5.jpg")

# NOTE: This is the code I used to extract the template images (from 5.jpg in the images directory)
#       leaving it around just in case it's needed in the future

# quarter_note_template_img = raw_image[244:254,164:177,:]
# half_note_template_img = raw_image[419:429,595:608,:]

# cv2.imwrite("./template_images/qtr_template.jpg", quarter_note_template_img)
# cv2.imwrite("./template_images/hlf_template.jpg", half_note_template_img)

# # resize for display
# qtr = cv2.resize(quarter_note_template_img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
# hlf = cv2.resize(half_note_template_img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

# cv2.imshow("qtr template", qtr)
# cv2.imshow("hlf template", hlf)
# cv2.waitKey(0)

quarter_note_template_img = cv2.imread("./template_images/qtr_template.jpg")
half_note_template_img = cv2.imread("./template_images/hlf_template.jpg")

# compute the scores from the template matching
qtr_scores = cv2.matchTemplate(raw_image, quarter_note_template_img, cv2.TM_CCOEFF_NORMED)
hlf_scores = cv2.matchTemplate(raw_image, half_note_template_img, cv2.TM_CCOEFF_NORMED)

# threshold said scores so that we get a 1 out any time there is an match in the image
# (the specififc threshold value is tuned by hand for this specific application)
low_thresh = 0.77
_, qtr_thresholded_scores = cv2.threshold(qtr_scores,thresh=low_thresh,maxval=1.0,type=cv2.THRESH_BINARY)
_, hlf_thresholded_scores = cv2.threshold(hlf_scores,thresh=low_thresh,maxval=1.0,type=cv2.THRESH_BINARY)

# get centroids of connected components (reject duplicates)
qtr_template_matches = cv2.connectedComponentsWithStats(np.uint8(255*qtr_thresholded_scores))[3]
hlf_template_matches = cv2.connectedComponentsWithStats(np.uint8(255*hlf_thresholded_scores))[3]
# remove "background label" (always appears in the centre of the image and is useless)
# these are now lists of image coordinate tuples of the top left corner of each match
qtr_template_matches = qtr_template_matches[1:]
hlf_template_matches = hlf_template_matches[1:]

def drawMatches(matches, template, image, color=(0,0,255)):
    for match in matches:
        r = round(match[1])
        c = round(match[0])
        tl = (c,r)
        br = (c + template.shape[1], r + template.shape[0])
        # add the rectangle
        cv2.rectangle(image, tl, br, color=color)

drawMatches(qtr_template_matches, quarter_note_template_img, raw_image, color=(0,0,255))
drawMatches(hlf_template_matches, half_note_template_img, raw_image, color=(0,255,0))

def getTemplateMatchCentroids(matches, template):
    for match in matches:
        r = round(match[1])
        c = round(match[0])
        centroid = (c + template.shape[1]//2, r + template.shape[0]//2)
        return centroid

# print the total number of matches for cli debugging
print(len(qtr_template_matches) + len(hlf_template_matches))

# NOTE: these arrays are the centroids of each match, use this for matching with the staff to 
#       determine horizontal order and pitch
qtr_template_match_centroids = getTemplateMatchCentroids(qtr_template_matches, quarter_note_template_img)
hlf_template_match_centroids = getTemplateMatchCentroids(hlf_template_matches, half_note_template_img)

interpretMusic.findLines("./images/5.jpg")

cv2.imshow("img",raw_image)
cv2.waitKey(0)