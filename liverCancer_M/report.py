# Function to generate report
from turtle import st


def generate_report(patient_info):
    st.title("Patient Report")
    # Display patient information
    st.write("Patient Information:")
    for key, value in patient_info.items():
        st.write(f"{key}: {value}")
    # Placeholder for liver cancer detection project output
    st.write("Liver Cancer Detection Project Output:")
    # Add code to display the output of your liver cancer detection project here
    # For example:
    st.write("Results:")
    st.write("Detected: Yes")
    st.write("Confidence Level: High")
