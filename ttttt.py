import cv2
import dlib
import pyttsx3
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time
import keras
from keras.utils.image_utils import img_to_array
import numpy as np
# INITIALIZING THE pyttsx3 SO THAT
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()


# SETTING UP OF CAMERA TO 1 YOU CAN
# EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(0)

# FACE DETECTION OR MAPPING THE FACE TO
# GET THE Eye AND EYES DETECTED
face_detector = dlib.get_frontal_face_detector()

# PUT THE LOCATION OF .DAT FILE (FILE FOR
# PREDECTING THE LANDMARKS ON FACE )
dlib_facelandmark = dlib.shape_predictor(
    "C:\\Users\\Phuoc\\Downloads\\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")


# FUNCTION CALCULATING THE ASPECT RATIO FOR
# THE Eye BY USING EUCLIDEAN DISTANCE FUNCTION


# MAIN LOOP IT WILL RUN ALL THE UNLESS AND
# UNTIL THE PROGRAM IS BEING KILLED BY THE USER


while True:
    null, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)

    for face in faces:
        # Dự đoán các điểm landmark trên khuôn mặt
        landmarks = dlib_facelandmark(gray_scale, face)
        # Lấy tọa độ của mắt trái và mắt phải
        # Lấy tọa độ của mắt trái và mắt phải
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # Tính toán góc quay của mắt
        angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        angle = math.degrees(angle)

        # Tính toán kích thước và tọa độ của hình vuông xung quanh mắt
        eye_width = abs(right_eye[0] - left_eye[0])
        eye_height = int(eye_width * 0.6)  # Chỉnh sửa tỷ lệ chiều cao của hình vuông tùy ý
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_eye = cv2.warpAffine(frame, rotation_matrix, frame.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Lấy hình vuông xung quanh mắt từ hình ảnh đã xoay
        eye_roi = rotated_eye[center[1] - eye_height // 2:center[1] + eye_height // 2,
                  center[0] - eye_width // 2:center[0] + eye_width // 2]

    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    key = cv2.waitKey(9)
    if key == 20:
        break



cap.release()
cv2.destroyAllWindows()