import joblib

data = joblib.load("cancer.pkl")

# print(data)
# Load the contents of the .pkl file
with open("cancer.pkl", "rb") as f:
    data = joblib.load(f)

# Now you can work with the loaded data
print(data)

