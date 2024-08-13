# keras39_cnn05_kaggle_bike.py 복사

# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv (카글 컴피티션 사이트)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, LSTM 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
path = 'C://ai5/_data/kaggle//bike-sharing-demand/'  

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)  # (10886, 11)
# print(test_csv.shape)   # (6493, 10)
# print(sampleSubmission.shape)   # (6493, 1)

########### x와 y를 분리
x  = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
# print(x)    # [10886 rows x 10 columns]

y = train_csv['count']
# print(y.shape)  # (10886,)

# print(x.shape)  # (10886, 8)

x = x.to_numpy()
x = x.reshape(10886, 4, 2)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=11)

# scaler = StandardScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델 구성   모래시계 모형은 안됨.
# model = Sequential()
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(2, 2, 2), padding='same'))
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
model.add(LSTM(21, return_sequences=True, input_shape=(4, 2), activation='relu')) # timesteps, features
model.add(LSTM(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
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

model.fit(x_train, y_train, epochs=1, batch_size=32,
          verbose=1, 
          validation_split=0.3, callbacks= [es, mcp])  
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

# print(test_csv.shape)   # (6493, 8)

# exit()

test_csv = test_csv.to_numpy()
test_csv = test_csv.reshape(6493, 4, 2)

# print(test_csv.shape)   # (6493, 2, 2, 2)

# exit()

y_submit = model.predict(test_csv)

# print(y_submit)
# print(y_submit.shape)   # (10886,)

# exit()  # 

sampleSubmission['count'] = y_submit
print(sampleSubmission)
print(sampleSubmission.shape)   #(10886,)

sampleSubmission.to_csv(path + "sampleSubmission_keras59_LSTM05_kaggle_bike_0813_1745.csv")

print("r2스코어 : ", r2)
print("로스 : ", loss)
# print("cout의 예측값 : ", )
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# if(gpus):
#     print("쥐피유 돈다!!!")
# else:
#     print("쥐피유 없다! xxxx")

# CPU: 걸린시간 :  9.84 초
# GPU: 걸린시간 :  38.11 초

# dnn 데이터 -> cnn데이터로 바꾸기
# r2스코어 :  0.2787635009423006 / 로스 :  24491.5625 / 걸린시간 :  73.04 초
# r2스코어 :  0.2985208837347578 / 로스 :  23820.650390625 / 걸린시간 :  72.23 초

# LSTM 모델 
# r2스코어 :  -0.24626100339053125 / 로스 :  42320.20703125 / 걸린시간 :  4.11 초
