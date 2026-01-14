import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib

# Load the classifier
clf = joblib.load("cancer.pkl")


# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    median_blurred = cv2.medianBlur(blurred, 3)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # DISPLAY

    return median_blurred


# Function to extract HOG features
def extract_hog_features(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fd = hog(thresh, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    return fd


# Function to predict tumor
def predict_tumor(image):
    preprocessed_image = preprocess_image(image)
    hog_features = extract_hog_features(preprocessed_image)
    prediction = clf.predict(np.array([hog_features], 'float64'))
    return prediction[0]


# def myclass(image):
#     pass

# Function to perform image segmentation and calculate tumor percentage
def segment_tumor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    #thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # DISPLAY
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DISPLA
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(image, contours, i, (255, 0, 0), 1)

    mask = np.zeros(image.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    a = cv2.drawContours(mask, [largest_areas[-3]], 0, (255, 255, 255, 255), -1)

    contours1, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tumor_area = 0
    tumor_contours = []
    for contour in contours1:
        area = cv2.contourArea(contour)
        if area > 500:  # Ignore small contours (noise)
            tumor_area += area
            tumor_contours.append(contour)

    total_area = image.shape[0] * image.shape[1]
    tumor_percentage = (tumor_area / total_area) * 100

    return tumor_percentage,total_area


def main():
    st.title("PRATIKSANTI : Safeguarding of  Lives With Cancer Detection ")


    st.write("Upload an image to detect tumor.")

    uploaded_file = st.file_uploader("Choose an image...", type=["bmp"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        #Predict tumor
        prediction = predict_tumor(image)
        if prediction == 1:
            st.error("Tumor detected!")
            #st.write("Calculating tumor percentage...")

            #Segment tumor and calculate percentage
            tumor_percentage, tumor_contours = segment_tumor(image)
            st.write(f"Tumor Percentage: {tumor_percentage:.2f}%")

            # Display cancer stage information based on tumor percentage
            if tumor_percentage < 10:
                st.write("Based on the tumor percentage, it may be an early stage of cancer.")
            elif 10 <= tumor_percentage < 50:
                st.write("Based on the tumor percentage, it may indicate an intermediate stage of cancer.")
            else:
                st.write("Based on the tumor percentage, it may be an advanced stage of cancer.")

            # Create mask for the tumor
            mask = np.zeros_like(preprocess_image(image))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # DISPLAY
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(image, markers)
            contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            largest_areas = sorted(contours, key=cv2.contourArea)
            a = cv2.drawContours(mask, [largest_areas[-3]], 0, (255, 255, 255, 255), -1)
            mask_3channel = cv2.merge([mask, mask, mask])
            #final_im = mask * image
            final_im = cv2.bitwise_and(image, mask_3channel)
            res_img = final_im.copy()
            kernel = np.ones((3, 3), dtype=np.uint8)
            final_result = cv2.erode(final_im, kernel, iterations=1)
            hsv = cv2.cvtColor(final_result, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            outp = cv2.subtract(final_im, res)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(image, markers)
            contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                if hierarchy[0][i][3] == -1:
                    cv2.drawContours(final_im, contours, i, (255, 0, 0), 1)

            mask = np.zeros(final_im.shape, np.uint8)
            largest_areas = sorted(contours, key=cv2.contourArea)
            b = largest_areas[:-1]
            for i in b:
                a = cv2.drawContours(mask, [i], -1, (255, 255, 255), -1)
            result = mask * res_img


            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(image, None)
            kp2, des2 = orb.detectAndCompute(mask, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            liver_match = cv2.drawMatches(image, kp1, mask, kp2, matches[:25], None, flags=2)

            st.image(mask, caption='Otsu Image', use_column_width=True)
            st.image(result, caption='Mask Image', use_column_width=True)

            st.image(liver_match, caption='Segmented Image', use_column_width=True)

            #################################################################

            median_blurred=preprocess_image(image)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)

            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # DISPLAY

            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DISPLAY

            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  # DISPLAY

            sure_fg = np.uint8(sure_fg)

            unknown = cv2.subtract(sure_bg, sure_fg)
            ret, markers = cv2.connectedComponents(sure_fg)

            markers = markers + 1

            markers[unknown == 255] = 0  # DISPLAY

            markers = cv2.watershed(image, markers)  # DISPLAY
            contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                if hierarchy[0][i][3] == -1:
                    cv2.drawContours(image, contours, i, (255, 0, 0), 1)

            mask = np.zeros(image.shape, np.uint8)
            largest_areas = sorted(contours, key=cv2.contourArea)
            a = cv2.drawContours(mask, [largest_areas[-3]], 0, (255, 255, 255, 255), -1)
            final_im = mask * image

            res_img = final_im.copy()

            kernel = np.ones((3, 3), dtype=np.uint8)
            final_result = cv2.erode(final_im, kernel, iterations=1)

            hsv = cv2.cvtColor(final_result, cv2.COLOR_BGR2HSV)

            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

            res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            outp = cv2.subtract(final_im, res)

            outp_gray = cv2.cvtColor(outp, cv2.COLOR_BGR2GRAY)

            ret, ct_thresh = cv2.threshold(outp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # DISPLAY

            # NOISE REMOVAL (OPTIONAL)
            kernel = np.ones((3, 3), np.uint8)

            opening = cv2.morphologyEx(ct_thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # DISPLAY

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

            markers = cv2.watershed(image, markers)  # DISPLAY

            # image, contours, hierarchy = cv2.findContours(markers.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            for i in range(len(contours)):
                if hierarchy[0][i][3] == -1:
                    cv2.drawContours(final_im, contours, i, (255, 0, 0), 1)

            mask = np.zeros(final_im.shape, np.uint8)
            largest_areas = sorted(contours, key=cv2.contourArea)
            b = largest_areas[:-1]
            for i in b:
                a = cv2.drawContours(mask, [i], -1, (255, 255, 255), -1)
            st.image(a, caption='tumor detected Image', use_column_width=True)


            # Highlight tumor in white and rest in black
            #tumor_highlighted = np.where(mask == 255, 255, 0).astype(np.uint8)

            # Display the preprocessed grayscale image with the tumor section highlighted
            #st.image(tumor_highlighted, caption='Preprocessed Grayscale Image with Tumor Section Highlighted', use_column_width=True, channels='GRAY')

            # Draw tumor contours on the original image
            #tumor_marked_image = image.copy()
            #cv2.drawContours(tumor_marked_image, tumor_contours, -1, (0, 255, 0), 2)

            # Display the marked tumor image
            #st.image(tumor_marked_image, caption='Tumor Marked Image', use_column_width=True)

        else:
            st.success("No tumor detected.")


if __name__ == '__main__':
    main()
