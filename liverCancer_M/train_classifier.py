import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
import joblib
import os


# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    median_blurred = cv2.medianBlur(blurred, 3)
    return median_blurred


# Function to extract HOG features
def extract_hog_features(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fd = hog(thresh, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    return fd


# Function to generate images and labels
def generate_images(folder, label):
    features = []
    labels = []

    for filename in os.listdir(folder):
        if filename.endswith(".bmp"):
            name = os.path.join(folder, filename)
            im = cv2.imread(name)
            if im is None:
                print(f"Error loading image: {name}")
                continue

            preprocessed_image = preprocess_image(im)
            hog_features = extract_hog_features(preprocessed_image)

            features.append(hog_features)
            labels.append(label)

    return features, labels


# Generate and save positive images
positive_folder = "new folder/Positive"
positive_features, positive_labels = generate_images(positive_folder, 1)

# Generate and save negative images
negative_folder = "new folder/Negative"
negative_features, negative_labels = generate_images(negative_folder, 0)

# Combine features and labels
hog_features = np.vstack((positive_features, negative_features))
labels = np.hstack((positive_labels, negative_labels))

# Create and train the classifier
clf = LinearSVC()
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "cancer.pkl", compress=3)

print("Classifier trained and saved successfully!")
