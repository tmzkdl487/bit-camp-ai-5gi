# keras39_cnn06_cancer.py 복사

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten,MaxPooling2D, LSTM  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (569, 30) (569,) <- 넘파이 데이터
# print(type(x))  # <class 'numpy.ndarray'>라고 나옴.

x = x.reshape(569, 10, 3)
x = x/255.



x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8, 
                                                    random_state= 3434)


# scaler = MaxAbsScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델링
# model = Sequential()
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(5, 3, 2), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
# model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Dropout(0.2))
# model.add(Dense(16, input_shape=(32,)))
# model.add(Dense(1))

#2. LSTM모델구성
model = Sequential()
model.add(LSTM(21, return_sequences=True, input_shape=(10, 3), activation='relu')) # timesteps, features
model.add(LSTM(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode ='min',
    patience = 10,
    restore_best_weights=True,
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

model.fit(x_train, y_train, epochs=10, batch_size=32,
verbose=1, validation_split=0.2, callbacks= [es, mcp])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
# print(y_pred[:20])
y_pred = np.round(y_pred)
# print(y_pred[:20])

accuracy_score = accuracy_score(y_test, y_pred)

print("로스 : ", loss[0])
print("ACC : ", round(loss[1], 3))   # 소수 3째자리 만들기
print("acc", accuracy_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# if(gpus):
#     print("쥐피유 돈다!!!")
# else:
#     print("쥐피유 없다! xxxx")

# CPU: 걸린시간 :  2.06 초
# GPU: 걸린시간 :  4.58 초

# dnn 데이터 -> cnn데이터로 바꾸기
# 로스 :  0.07301134616136551 / ACC :  0.886 / acc 0.8859649122807017 / 걸린시간 :  4.59 초

# LSTM 모델 
# 로스 :  0.0985855907201767 / ACC :  0.86 / acc 0.8596491228070176