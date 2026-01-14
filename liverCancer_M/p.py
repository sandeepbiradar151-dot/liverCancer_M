import streamlit as st

def patient_information_form():
    st.subheader("Patient Information")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=150, step=1)
    gender = st.radio("Gender", ("Male", "Female", "Other"))
    # Add more fields as needed
    return name, age, gender

def main():
    st.title("Cancer Detection Report Generator")
    name, age, gender = patient_information_form()
    st.write(f"Name: {name}, Age: {age}, Gender: {gender}")

if __name__ == "__main__":
    main()
