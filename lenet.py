# Đây là file định nghĩa model(mạng) và các layer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Khởi tạo model
        model = Sequential()
        input_shape = (height, width, depth)#(28, 28, 1)

        # sử dụng 'channels-first' để input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # Thứ nhất: CONV => RELU => POOL layers
        model.add(Conv2D(6, (5, 5), padding='same', input_shape=input_shape))#(28, 28, 6)
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))#(14, 14, 6)

        # Thứ hai: CONV => RELU => POOL layers
        model.add(Conv2D(16, (5, 5), padding='same'))#(14, 14, 16)
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))#(7, 7, 16)

        # Làm phẳng rồi đưa vào lớp FC => RELU layers
        model.add(Flatten())# 784
        model.add(Dense(150))# 150
        model.add(Activation('relu'))

        # Softmax classifier
        model.add(Dense(classes))#2
        model.add(Activation('softmax'))

        # Trả về model (Mạng CNN lenet)
        return model
