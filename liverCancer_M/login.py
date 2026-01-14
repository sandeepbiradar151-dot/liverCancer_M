# login_page.py

import streamlit as st

def main():
    st.title("Login")

    # Input fields for username/email and password
    username = st.text_input("Username/Email")
    password = st.text_input("Password", type="password")

    # Submit button for the form
    if st.button("Login"):
        # Check if both username/email and password are filled
        if username and password:
            # Placeholder authentication logic
            if authenticate(username, password):
                st.success("Login successful")
            else:
                st.error("Invalid username/email or password.")
        else:
            st.error("Please enter both username/email and password.")



# Function to handle authentication (placeholder logic)
def authenticate(username, password):
    # Replace this placeholder logic with actual authentication logic
    return username == "user" and password == "pass"

if __name__ == "__main__":
    main()
