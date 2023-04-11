import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot

st.header("Generate ASCII images using GAN")
st.write("Choose any image and get corresponding ASCII art:")

uploaded_file = st.file_uploader("Choose an image...")