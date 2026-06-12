import streamlit as st
st.title("My first streamlit application")
name=st.text_input("Enter your name")
n1=st.number_input("Enter your age")
if st.button("Submit"):
         st.write(f"Your name is {name},age is  {n1}")
         
