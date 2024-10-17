# keras41_ImageDataGernerator5_rps.py 복사

# ACC 1.0 만들기

# keras41_ImageDataGenerator1.py
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255)

path_train = './_data/image/rps/'

start_time1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(80, 80), 
    batch_size=2520, 
    class_mode='categorical', # 다중분류 - 원핫도 되서 나와욤.
    # class_mode='binary',    # 이중분류
    # color_mode='sparse',     # 다중분류
    # class_mode='None',       # y값이 없다!!!
    
    # color_mode='grayscale',
    color_mode='rgb',
    shuffle=True,
)   # Found 2520 images belonging to 3 classes.
# print(xy_train[0][1]) -> 값
# [1. 1. 2. 2. 1. 2. 0. 1. 2. 0. 0. 0. 2. 2. 2. 0. 1. 0. 0. 0. 1. 2. 1. 0.  0. 1. 0. 0. 1. 1.]

# print(xy_train[0][0].shape) # (30, 100, 100, 1)
# print(xy_train[0][0].shape) # (30, 100, 100, 3)

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, 
                                                    shuffle= True,
                                                    random_state=666)

# print(xy_train[0][0].shape) # (2520, 80, 80, 3) <- 전에 거.
# print(xy_train[0][1].shape) # (2520, 3)

# print(x_train.shape, x_test.shape)  # (2016, 80, 80, 3) (504, 80, 80, 3)

# exit()

x_train = x_train.reshape(2016, 80, 80*3)
x_test = x_test.reshape(504, 80, 80*3)

# exit()

end_time1 = time.time()

#2. 모델
# model = Sequential()
# model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(80, 80, 3)))
# model.add(MaxPooling2D(2, 2))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(2, 2))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(2, 2))                        
# model.add(Flatten())                  
# model.add(Dense(64, activation='relu'))  
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))  
# model.add(Dropout(0.5))
# model.add(Dense(3, activation='softmax'))

#2. LSTM모델구성
model = Sequential()
model.add(LSTM(21, return_sequences=True, input_shape=(80, 80*3), activation='relu')) # timesteps, features
model.add(LSTM(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])  
start_time2 = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
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

path = './_save/keras35/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k41_rps', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( 
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=10, batch_size=10,
          validation_split=0.3, verbose=1, callbacks=[es, mcp])

end_time2 = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)  

y_pred = model.predict(x_test)

print("59_rps_로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("데이터 걸린시간 : ", round(end_time1 - start_time1, 2), "초")
print("걸린시간 : ", round(end_time2 - start_time2, 2), "초")

# ACC 1.0 만들기
# 로스는 :  8.843438263284042e-05 / ACC :  1.0 / 데이터 걸린시간 :  5.76 초 / 걸린시간 :  34.5 초

# LSTM 모델 
# 59_rps_로스는 :  5.500619411468506 / ACC :  0.341 / 데이터 걸린시간 :  5.87 초 / 걸린시간 :  180.29 초
