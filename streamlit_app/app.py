import streamlit as st
import numpy as np
import cv2
import pandas as pd
#from deepface import DeepFace
from tensorflow import keras
from keras.models import model_from_json
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode


st.title("Emotional recognition :scream_cat:")

# section for photo
uploaded_file = st.file_uploader("Choose your photo...")





#live time video
st.header("Live Camera")

# run = st.button('Run')
# FRAME_WINDOW = st.image([])

# else:
#         st.write('Stopped')

vid = cv2.VideoCapture(1)

st.title( 'Using Mobile Camera with Streamlit' )
frame_window = st.image( [] )
take_picture_button = st.button( 'Take Picture' )

while True:
    got_frame , frame = vid.read()
    frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
    if got_frame:
        frame_window.image(frame)

    if take_picture_button:
        # Pass the frame to a model
        # And show the output here...
        break

vid.release()

# # load model
# emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# # load json and create model
# json_file = open('emotion_model1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# classifier = model_from_json(loaded_model_json)

# # load weights into new model
# classifier.load_weights("emotion_model1.h5")

# #load face
# try:
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# except Exception:
#     st.write("Error loading cascade classifiers")

# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# class Faceemotion(VideoTransformerBase):
#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         #image gray
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(
#             image=img_gray, scaleFactor=1.3, minNeighbors=5)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(img=img, pt1=(x, y), pt2=(
#                 x + w, y + h), color=(255, 0, 0), thickness=2)
#             roi_gray = img_gray[y:y + h, x:x + w]
#             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#             if np.sum([roi_gray]) != 0:
#                 roi = roi_gray.astype('float') / 255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)
#                 prediction = classifier.predict(roi)[0]
#                 maxindex = int(np.argmax(prediction))
#                 finalout = emotion_dict[maxindex]
#                 output = str(finalout)
#             label_position = (x, y)
#             cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         return img





# # /Users/roma/Library/Python/3.11/lib/python/site-packages 