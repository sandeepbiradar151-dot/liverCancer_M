import streamlit as st
from login_page import main as login_page
from sign_up_page import main as sign_up


# Function to display the main page
def main_page():
    st.title("Pratiksanti : Safeguarding of lives for liver cancer")

    # Display image
    st.image("C:\\Users\\USER\\Desktop\\download.jpeg", caption="Image Caption")

    # Display paragraph
    st.write("Is life worth living? It all depends on the liver."
             "-William James")

# Set up an empty placeholder to be filled dynamically
placeholder = st.empty()

# Display the main page initially
main_page()

# Check if the button to navigate to the login page is clicked
if placeholder.button("Go to Login"):
    # Clear the placeholder
    placeholder.empty()
    # Display the login page
    login_page()

