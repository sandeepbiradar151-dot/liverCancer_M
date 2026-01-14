import cv2
import numpy as np

ct_liver = cv2.imread("Dataset/in/")
img = ct_liver.copy()  # DISPLAY
#img2 = ct_liver.copy()

# APPLY BLUR
img = cv2.medianBlur(img, 3)  # DISPLAY

# CONVERT TO GRAY SCALE
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# APPLY THRESHOLD NOT BINARY ==> APPLY OTSU'S METHOD
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # DISPLAY

# NOISE REMOVAL (OPTIONAL)
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # DISPLAY

sure_bg = cv2.dilate(opening, kernel, iterations=3)  # DISPLAY

# DISTANCE TRANSFORM
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DISPLAY

ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  # DISPLAY

sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)  # DISPLAY

# CREATING MARKERS => 3 STEPS

# 1. GETTING MARKERS

ret, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1

markers[unknown == 255] = 0  # DISPLAY

markers = cv2.watershed(img, markers)  # DISPLAY

# image, contours, hierarchy = cv2.findContours(markers.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(ct_liver, contours, i, (255, 0, 0), 1)

mask = np.zeros(ct_liver.shape, np.uint8)
largest_areas = sorted(contours, key=cv2.contourArea)
a = cv2.drawContours(mask, [largest_areas[-3]], 0, (255, 255, 255, 255), -1)

print("ok")