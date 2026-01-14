# main_page.py
import streamlit as st
from login import loginnn_form,signup_form
def main():
    st.title("Your Project Title")
    # Add project image
    st.image("E:\liverCancer_M\IMG.jpg ", caption="Project Image", use_column_width=True)
    # Add login button
    login_button = st.button("Login")
    if login_button:
        loginnn_form()
    else:
        signup_form()

if __name__ == "__main__":
    main()
