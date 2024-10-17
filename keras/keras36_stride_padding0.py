# keras35_cnn3_연산량계산_안했다.py 복사

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten  # Flatten = 나라시, 평평하게 하다.
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import time


# #1. 데이터
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) <- 흑백이미지는 맨 뒤에 1이 생략되어서 오기 때문에 (60000, 28, 28, 1)인 것이다.
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000, 10)

#2. 모델
model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=(10, 10, 1),
                 strides=2,
                 padding='same'
                #  padding='valid'    # 디폴트는 발리드
                 ))   # 27, 27, 10
model.add(Conv2D(filters=20, kernel_size=(3,3),
          strides=1,
          padding='valid'))        # 8, 8, 20       
 
# model.add(Conv2D(15, (4,4)))    

# model.add(Flatten())    # 4차원 데이터를 2차원 데이터로 만들어 줌.

# model.add(Dense(units=8))   # N, 22x22x15 = 7260
# model.add(Dense(units=9, input_shape=(8,)))
#                         # shape= (batch_size, input_dim)
# model.add(Dense(10, activation='softmax'))

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 27, 27, 10)        50

#  conv2d_1 (Conv2D)           (None, 25, 25, 20)        1820

#  conv2d_2 (Conv2D)           (None, 22, 22, 15)        4815

#  dense (Dense)               (None, 22, 22, 8)         128

#  dense_1 (Dense)             (None, 22, 22, 9)         81

# =================================================================
# Total params: 6,894
# Trainable params: 6,894
# Non-trainable params: 0
# _________________________________________________________________

# Flatten하고 찍은 써머리
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 27, 27, 10)        50

#  conv2d_1 (Conv2D)           (None, 25, 25, 20)        1820

#  conv2d_2 (Conv2D)           (None, 22, 22, 15)        4815

#  flatten (Flatten)           (None, 7260)              0  # <- 모양만 바꿔줘서 연산량이 없음.

#  dense (Dense)               (None, 8)                 58088

#  dense_1 (Dense)             (None, 9)                 81

# =================================================================
# Total params: 64,854
# Trainable params: 64,854
# Non-trainable params: 0
# _________________________________________________________________

# strides=2하고 찍은 서머리. 아웃풋 반토막 남. strides=디폴트는 1이다. 
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 14, 14, 10)        50        

#  conv2d_1 (Conv2D)           (None, 12, 12, 20)        1820      

#  conv2d_2 (Conv2D)           (None, 9, 9, 15)          4815      

#  flatten (Flatten)           (None, 1215)              0

#  dense (Dense)               (None, 8)                 9728

#  dense_1 (Dense)             (None, 9)                 81

#  dense_2 (Dense)             (None, 10)                100

# =================================================================
# Total params: 16,594
# Trainable params: 16,594
# Non-trainable params: 0
# _________________________________________________________________

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# start_time = time.time()

# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',
#     patience = 10,
#     restore_best_weights= True
# )

# model.fit(x_train, y_train, epochs=2, batch_size= 128, verbose=1, validation_split=0.2)

# end_time = time.time()

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test, verbose=1)

# y_pred = model.predict(x_test)
# r2 = r2_score(y_test, y_pred)

# print('r2 스코어 : ', r2)
# print("로스는 : ", loss[0])
# print("ACC : ", round(loss[1], 3))
# print("걸린시간 : ", round(end_time - start_time, 2), "초")

# # # CPU: 걸린시간: r2 스코어 :  0.7506777688474371 / 로스는 :  0.5591500401496887 / ACC :  0.852 / 걸린시간 :  23.05 초
# # # GPU: 걸린시간: r2 스코어 :  0.7435134250531441 / 로스는 :  0.536513090133667 / ACC :  0.847 / 걸린시간 :  4.93 초


# model = Sequential()
# model.add(Conv2D(10, (2,2), input_shape=(28, 28, 1)))  
# model.add(Conv2D(filters=20, kernel_size=(3,3)))   
# model.add(Conv2D(15, (4,4)))   
# model.add(Flatten())   
# model.add(Dense(units=8)) 
# model.add(Dense(units=9, input_shape=(8,)))                    
# model.add(Dense(10, activation='softmax'))

