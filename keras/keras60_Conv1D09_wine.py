# keras39_cnn09_wine.py 복사

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPool1D   
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
dataset = load_wine()

x = dataset.data
y = dataset.target
# print(x.shape, y.shape) # (178, 13) (178,)

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape)  # (178, 3)

# print(x.shape)  # (178, 13)

x = x.reshape(178, 13, 1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9,
                                                    random_state=666,
                                                    stratify=y)
scaler = RobustScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler

#2. 모델
# model = Sequential()
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(13, 1, 1), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
# model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu', input_shape=(32,)))
# model.add(Dense(1))

#2. Conv1D 모델
model = Sequential()
model.add(Conv1D(10, kernel_size=2, input_shape=(13, 1))) # timesteps, features
model.add(Conv1D(10, 2))
model.add(Flatten())
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights= True
)

########################### mcp 세이프 파일명 만들기 시작 ################
import datetime 
date = datetime.datetime.now()
print(date) # 2024-07-26 16:51:36.578483
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date) # 0726 / 0726_1654
print(type(date))

path = './_save/keras32/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k32_', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1, validation_split=0.2, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print("로스 : ", round(loss[0], 3))
print("ACC : ", round(loss[1], 3))  # 반올림
print("걸린시간 : ", round(end_time - start_time,2), "초")

# [실습] stratify=y을 넣고 돌려보기.
# ACC :  ACC :  0.389

# 그냥 
# 로스는 :  [0.10185517370700836, 1.0] / ACC :  1.0

# [실습] MinMaxScaler 넣고 점수 갱신해보기
# 로스는 :  [0.9439105987548828, 0.9444444179534912] / ACC :  0.944

# [실습] StandardScaler 스켈링하고
# 로스는 :  [0.004755509551614523, 1.0] / ACC :  1.0

# [실습] MaxAbsScaler 스켈링하고 돌려보기.
# 로스 :  [1.0360684394836426, 0.8888888955116272] / ACC :  0.889

# [실습] RobustScaler 스켈링하고 돌려보기.
# 로스 :  [0.434501051902771, 0.8888888955116272] / ACC :  0.889

# 세이브 가중치
# 로스 :  [0.02588886208832264, 1.0]
# ACC :  1.0

# 로스 :  [0.02588886208832264, 1.0]
# ACC :  1.0

# 드롭아웃 하고 나서
# 로스 :  [0.3165428042411804, 0.9444444179534912] / ACC :  0.944 / 걸린시간 :  5.01 초

# dnn 데이터 -> cnn데이터로 바꾸기
# 로스 :  [0.2222222089767456, 0.6666666865348816] / ACC :  0.667 /걸린시간 :  24.27 초
# 로스 :  0.222 / ACC :  0.667 / 걸린시간 :  22.38 초
# 로스 :  0.222 / ACC :  0.667 / 걸린시간 :  20.26 초

# 모델 Conv1D 
# 로스 :  0.222 / ACC :  0.667 / 걸린시간 :  46.41 초