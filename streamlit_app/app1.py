import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot

st.header("Emotional recognition :scream_cat:")
st.write("Choose any image and get your emoeion")
uploaded_file = st.file_uploader("Choose an image...")