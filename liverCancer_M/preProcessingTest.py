# -*- coding: utf-8 -*-

import cv2
import numpy as np

def printValue(event,x,y,flags,param):
    print(im_gray1[y,x])

test= True
i=1
labels=[]
features=[]
while test:
    
    name="New folder/Positive/a ("+str(i)+").bmp"
    im = cv2.imread(name)  #import image
    im = cv2.GaussianBlur(im, (9, 9), 0)
    im = cv2.medianBlur(im,3)
    im_gray1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    
    lower_value = np.array([60])
    upper_value = np.array([190])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(im_gray1, lower_value, upper_value)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(im_gray1,im_gray1, mask= mask)
    u= res[res[:,:] >0 ]
#    print(u.mean())
#    print(u.max())
#    print(u.min())

    res = cv2.medianBlur(res,11)
    res = cv2.GaussianBlur(res, (9, 9), 0)
    
    # Threshold the image
    res[res[:,:] < (u.mean()-30) ]=0
    res[res[:,:] > (u.mean()+20) ]=255

    # Find contours in the image
    _, ctrs, _ = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaVal=[ cv2.contourArea(ctr) for ctr in ctrs]
    maxArea=np.max(areaVal)
    ind=np.argmax(areaVal)
#    print(ind)
    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    
    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.

    mask = np.zeros_like(im_gray1) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, ctrs, ind, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(im_gray1) # Extract out the object and place into output imag
    out[mask == 255] = im_gray1[mask == 255]
    lower_value = np.array([60])
    upper_value = np.array([200])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(out, lower_value, upper_value)
    # Bitwise-AND mask and original image
    out = cv2.bitwise_and(out,out, mask= mask)
    v= out[out[:,:] >0 ]
#    print(v.mean())

    out[out[:,:] < (u.mean()) ]=255
    out[out[:,:] > (u.mean()+10) ]=0
    out = cv2.medianBlur(out,5)

    imq = cv2.resize(im,(512,512))
    cv2.imshow("TestImageGry", res)
    cv2.imshow("TestImage", imq)
    cv2.imshow("TestImageg", out) 
    cv2.setMouseCallback("TestImage",printValue)
    while(1):
        cv2.imshow("TestImageg", out) 
        cv2.imshow("TestImageGry", res)
        cv2.imshow("TestImage", imq)

        if cv2.waitKey(20) & 0xFF == 27:
            break
    i=i+1
    print(i)
    if i is 32:
        test=False   

print("Done!!!!!")
        
cv2.waitKey()
cv2.destroyAllWindows()