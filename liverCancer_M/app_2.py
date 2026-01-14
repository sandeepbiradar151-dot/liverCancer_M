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


# Function to perform image segmentation and calculate tumor percentage
def segment_tumor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # DISPLAY
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # DISPLAY
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DISPLA
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  # DISPL
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    contours, _ = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (tumor)
    tumor_area = 0
    tumor_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > tumor_area:
            tumor_area = area
            tumor_contour = contour

    total_area = image.shape[0] * image.shape[1]
    tumor_percentage = (tumor_area / total_area) * 100

    return tumor_percentage, tumor_contour


def main():
    st.title("Tumor Detection App")

    st.write("Upload an image to detect tumor.")

    uploaded_file = st.file_uploader("Choose an image...", type=["bmp"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict tumor
        prediction = predict_tumor(image)
        if prediction == 1:
            st.error("Tumor detected!")
            st.write("Calculating tumor percentage...")

            # Segment tumor and calculate percentage
            tumor_percentage, _ = segment_tumor(image)
            st.write(f"Tumor Percentage: {tumor_percentage:.2f}%")

            # Display the preprocessed image
            preprocessed_image = preprocess_image(image)
            st.image(preprocessed_image, caption='Preprocessed Image', use_column_width=True)

            # Display the segmented image
            _, segmented_image = segment_tumor(image)
            st.image(segmented_image, caption='Segmented Image', use_column_width=True)

        else:
            st.success("No tumor detected.")


if __name__ == '__main__':
    main()
