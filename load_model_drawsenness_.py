# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:12:01 2022

@author: sanau
"""
from datetime import datetime
from packaging import version
import os
import cv2
#from google.colab.patches import cv2_imshow
import numpy as np
from playsound import playsound
from PIL import Image, ImageDraw
import face_recognition
import keras
import time
import pyttsx3
# from pygame import mixer

# INITIALIZING THE pyttsx3 SO THAT
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

eye_model = keras.models.load_model('Drowsiness_model.h5')

# webcam frame is inputted into function
def eye_cropper(frame):
    # create a variable for the facial feature coordinates
    facial_features_list = face_recognition.face_landmarks(frame)

    # create a placeholder list for the eye coordinates
    # and append coordinates for eyes to list unless eyes
    # weren't found by facial recognition
    try:
        eye = facial_features_list[0]['left_eye']
    except:
        try:
            eye = facial_features_list[0]['right_eye']
        except:
            return

    # establish the max x and y coordinates of the eye
    x_max = max([coordinate[0] for coordinate in eye])
    x_min = min([coordinate[0] for coordinate in eye])
    y_max = max([coordinate[1] for coordinate in eye])
    y_min = min([coordinate[1] for coordinate in eye])

    # establish the range of x and y coordinates
    x_range = x_max - x_min
    y_range = y_max - y_min

    # in order to make sure the full eye is captured,
    # calculate the coordinates of a square that has a
    # 50% cushion added to the axis with a larger range and
    # then match the smaller range to the cushioned larger range
    if x_range > y_range:
        right = round(.5 * x_range) + x_max
        left = x_min - round(.5 * x_range)
        bottom = round((((right - left) - y_range)) / 2) + y_max
        top = y_min - round((((right - left) - y_range)) / 2)
    else:
        bottom = round(.5 * y_range) + y_max
        top = y_min - round(.5 * y_range)
        right = round((((bottom - top) - x_range)) / 2) + x_max
        left = x_min - round((((bottom - top) - x_range)) / 2)

    # crop the image according to the coordinates determined above
    cropped = frame[top:(bottom + 1), left:(right + 1)]


    # resize the image
    cropped = cv2.resize(cropped, (128, 128))
    image_for_prediction = cropped.reshape(-1, 128, 128, 3)

    return image_for_prediction


# initiate webcam
cap = cv2.VideoCapture(0)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
if not cap.isOpened():
    raise IOError('Cannot open webcam')


# create a while loop that runs while webcam is in use
not_drowsy_last_time = time.time()
drowsy_time = 0
while True:

    # capture frames being outputted by webcam
    null, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # function called on the frame
    image_for_prediction = eye_cropper(frame)
    try:
        image_for_prediction = image_for_prediction/255.0
    except:
        continue
    # get prediction from model
    prediction = eye_model.predict(image_for_prediction)
    print(prediction)
    if prediction <= 0.5:
        drowsy_time = time.time() - not_drowsy_last_time
    else:
        not_drowsy_last_time = time.time()
        drowsy_time = 0

    if drowsy_time > 3:
        cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
        cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

        # CALLING THE AUDIO FUNCTION OF TEXT TO
        # AUDIO FOR ALERTING THE PERSON
        engine.say("Alert!!!! WAKE UP DUDE")
        engine.runAndWait()
    print(drowsy_time)




    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()