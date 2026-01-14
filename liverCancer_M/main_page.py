# main_page.py

import streamlit as st

def main():
    st.title("Liver Cancer Detection")

    # Display image
    st.image("E:/liverCancer_M/IMG.jpg", caption="Image Caption")

    # Display paragraph
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
             "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")

    # Login button to navigate to login page
    if st.button("Login"):
        st.markdown("[Login](?page=login_page)")

if __name__ == "__main__":
    main()
