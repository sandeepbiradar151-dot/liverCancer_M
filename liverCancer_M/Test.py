import cv2
import joblib
from skimage.feature import hog
import numpy as np

clf = joblib.load("cancer.pkl")
test= True
i=1
while test:
    
    name="New folder/Positive/a ("+str(i)+").bmp"
    im = cv2.imread(name)
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

    res = cv2.medianBlur(res,11)
    res = cv2.GaussianBlur(res, (9, 9), 0)
    
    # Threshold the image

    res[res[:,:] < (u.mean()-30) ]=0
    res[res[:,:] > (u.mean()+20) ]=255

    # Find contours in the image
    ctrs, _ = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    fd = hog(out.reshape((512, 512)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    nbr = clf.predict(np.array([fd], 'float64'))
    print(str(nbr[0]))
    if str(nbr[0])=="true":
        out = cv2.medianBlur(out,5)
        _, ctrs, _ = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for ctr in ctrs:
            x,y,w,h = cv2.boundingRect(ctr)
            if x<250:
                im=cv2.drawContours(im, [ctr], 0, (0,0,255), -1)
        cv2.imshow("TestImage", im)
        cv2.waitKey()
        
    else:
        cv2.imshow("TestImage", im)
        cv2.waitKey()
    i=i+1
    print(i)
    if i is 32:
        test=False
        
print("whats up now")
cv2.waitKey()
cv2.destroyAllWindows()