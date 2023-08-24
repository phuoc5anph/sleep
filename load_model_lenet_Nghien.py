import threading
import cv2
import dlib
import pyttsx3
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time
import keras
from keras.utils.image_utils import img_to_array
import numpy as np
import math

def calculate_angle(left_eye, right_eye):
    # Tính toán khoảng cách theo trục X giữa hai mắt
    delta_x = right_eye[0] - left_eye[0]

    # Tính toán khoảng cách theo trục Y giữa hai mắt
    delta_y = right_eye[1] - left_eye[1]

    # Tính toán góc xoay dựa trên khoảng cách theo trục X và Y
    angle = math.degrees(math.atan2(delta_y, delta_x))

    return angle

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Tạo ma trận xoay
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Thực hiện xoay ảnh
    rotated_image = cv2.warpAffine(image, matrix, (width, height))

    return rotated_image
# Dùng để phát âm thanh cảnh báo
engine = pyttsx3.init()

# Hàm thực thi câu lệnh engine.say() trong một luồng riêng biệt
def speak_alert():
    if not engine._inLoop:
        engine.say("Alert!!!! WAKE UP DUDE")
        engine.runAndWait()

model = keras.models.load_model("lenet2.hdf5")

# Khởi tạo đối tượng camera để láy hình ảnh từ camera C:/Users/Phuoc/Pictures/Camera Roll/TGMT5.mp4
cap = cv2.VideoCapture("C:/Users/Phuoc/Pictures/Camera Roll/TGMT5.mp4")

# Bộ phát hiện khuôn mặt
face_detector = dlib.get_frontal_face_detector()

# Bộ láy các đặt trưng trên khuôn mặt
dlib_facelandmark = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")



# Thời gian không ngủ gật cuối cùng
not_drowsy_last_time = time.time()
# Số thời gian đã ngủ gật
drowsy_time = 0


while True:
    # Láy hình ảnh từ camera
    null, frame = cap.read()
    # Chuyển ảnh sang độ xám
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_scale)
    if len(faces) > 0:
        first_face = face_detector(gray_scale)[0]
        face_landmarks1 = dlib_facelandmark(gray_scale, first_face)
        # Lấy tọa độ của mắt trái và mắt phải
        left_eye = (face_landmarks1.part(36).x, face_landmarks1.part(36).y)
        right_eye = (face_landmarks1.part(45).x, face_landmarks1.part(45).y)

        # Tính toán góc xoay dựa trên tọa độ mắt
        angle = calculate_angle(left_eye, right_eye)
        # Xoay khung hình theo góc
        gray_scale = rotate_image(gray_scale, angle)
        # Phát hiện khuôn mặt
    faces = face_detector(gray_scale)



    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)

        x_max = face_landmarks.part(39).x
        x_min = face_landmarks.part(36).x
        y_max = face_landmarks.part(41).y
        y_min = face_landmarks.part(37).y

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

        cv2.rectangle(frame, (left, bottom), (right, top), (255, 255, 0), 1)

        # Cắt ảnh chỉ chứa mắt phù hợp với mô hình
        cropped = gray_scale[top:(bottom + 1), left:(right + 1)]
        #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cropped = cv2.resize(cropped, (28, 28))
        # Cân bằng độ tương phản để phù hợp với mô hình
        cropped = cv2.equalizeHist(cropped)
        cropped = cropped.astype("float") / 255.0
        cropped = img_to_array(cropped)

        # Chuyển ảnh sang dạng (1, height, width)
        image_for_prediction = np.expand_dims(cropped, axis=0)

        # Cropped = cv2.convertScaleAbs(cropped)

        # Dự đoán trạng thái nghủ gật
        prediction = model.predict(image_for_prediction)
        if prediction[0][1] < 0.5:
            drowsy_time = time.time() - not_drowsy_last_time
        else:
            not_drowsy_last_time = time.time()
            drowsy_time = 0
        print([format(prediction[0][1], ".4f"), drowsy_time])
        # Nếu thời gian ngủ gật vượt quá 3 thì cảnh báo
        if drowsy_time > 2:
            cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
            cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

            # Phát ra âm thanh
            #engine.say("Alert!!!! WAKE UP DUDE")
            #engine.runAndWait()
            # Tạo một thread mới để thực hiện chức năng alert
            alert_thread = threading.Thread(target=speak_alert)
            alert_thread.daemon = True
            alert_thread.start()




    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    key = cv2.waitKey(9)
    if key == 20:
        break


cap.release()
cv2.destroyAllWindows()