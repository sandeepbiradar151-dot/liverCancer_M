import streamlit as st

def main():
    st.title("Sign Up Page")

    # Input fields for username, email, and password
    with st.form(key='signup_form'):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        # Submit button for the form
        submitted = st.form_submit_button("Sign Up")

        # Check if the form is submitted
        if submitted:
            # Check if all fields are filled
            if username and email and password:
                # Call a function to handle sign-up and store data
                sign_up(username, email, password)
            else:
                st.error("Please fill in all fields.")

# Function to handle sign-up and store data
def sign_up(username, email, password):
    # Placeholder data storage (in-memory)
    users = []

    # Add user data to the list
    users.append({"username": username, "email": email, "password": password})

    # Display success message
    st.success("Sign up successful")
    st.write("Username:", username)
    st.write("Email:", email)
    # Note: In a real application, you would save user data to a database

if __name__ == "__main__":
    main()
