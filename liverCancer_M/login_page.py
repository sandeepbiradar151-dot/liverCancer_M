# login_page.py

import streamlit as st

def main():
    st.title("Login")

    # Form for username/email and password
    with st.form(key='login_form'):
        username = st.text_input("Username/Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            # Check if all fields are filled
            if username and password:
                # Call a function to handle sign-up and store data
                authenticate(username, password)
            else:
                st.error("Please fill in all fields.")
    # If sign-up is clicked, navigate to the sign-up page
    st.write("Don't have an account? ", end="")
    st.markdown("[Sign Up](http://localhost:8501/sign_up_page)")

def authenticate(username, password):
    if username == "user" and password == "pass":
        st.write("Login successfull")
if __name__ == "__main__":
    main()
