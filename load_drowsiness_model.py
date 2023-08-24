import cv2
import dlib
import pyttsx3
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time
import keras
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

eye_model = keras.models.load_model('Drowsiness_model.h5')

not_drowsy_last_time = time.time()
drowsy_time = 0


while True:
    null, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)

        x_max = face_landmarks.part(39).x
        x_min = face_landmarks.part(36).x
        y_max = face_landmarks.part(41).y
        y_min = face_landmarks.part(37).y

        #cv2.line(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
        # establish the range of x and y coordinates
        x_range = x_max - x_min
        y_range = y_max - y_min
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


        cv2.rectangle(frame, (top, bottom), (left, right), (255, 255, 0), 1)

        cropped = frame[top:(bottom + 1), left:(right + 1)]

        # resize the image
        cropped = cv2.resize(cropped, (128, 128))
        #cropped = cv2.convertScaleAbs(cropped)
        image_for_prediction = cropped.reshape(-1, 128, 128, 3)





        try:
            image_for_prediction = image_for_prediction / 255.0
        except:
            continue

        prediction = eye_model.predict(image_for_prediction)

        if prediction < 0.5:
            drowsy_time = time.time() - not_drowsy_last_time
        else:
            not_drowsy_last_time = time.time()
            drowsy_time = 0
        print(drowsy_time)

        if drowsy_time > 2:
            cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
            cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

            # CALLING THE AUDIO FUNCTION OF TEXT TO
            # AUDIO FOR ALERTING THE PERSON
            engine.say("Alert!!!! WAKE UP DUDE")
            engine.runAndWait()




    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    key = cv2.waitKey(9)
    if key == 20:
        break


cap.release()
cv2.destroyAllWindows()