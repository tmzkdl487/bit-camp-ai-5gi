# keras39_cnn03_diabetse.py 복사

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Conv1D, MaxPool1D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import sklearn as sk
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target 

# print(x)    
# print(y)
# print(x.shape, y.shape) # (442, 10) (442,)

x = x.reshape(442, 5, 2, 1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=8989)

scaler = MaxAbsScaler() # MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

print(x_train.shape)    # (353, 5, 2, 1)

x_train = x_train.reshape(353, 5, 2)  

# exit()

# 2. 모델 구성
# model = Sequential()
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(5, 2, 1), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
# model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Dropout(0.2))
# model.add(Dense(16, input_shape=(32,)))
# model.add(Dense(1))

#2. Conv1D 모델
model = Sequential()
model.add(Conv1D(10, kernel_size=2, input_shape=(5, 2))) # timesteps, features
model.add(Conv1D(10, 2))
model.add(Flatten())
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
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

model.fit(x_train, y_train, epochs=1000, batch_size=32, 
          verbose=1, validation_split=0.2,
          callbacks= [es, mcp]
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print ("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# if(gpus):
#     print("쥐피유 돈다!!!")
# else:
#     print("쥐피유 없다! xxxx")

# CPU: 걸린시간 :  1.02 초
# GPU: 걸린시간 :  1.78 초

# 그냥
# 로스 :  3531.4384765625 / r2스코어 :  0.4119794459086601

#[실습] MinMaxScaler 스켈링하고 돌려보기.
# 로스 :  3663.0 / r2스코어 :  0.39007311667484146

# [실습] StandardScaler 스켈링하고 돌려보기.
# 로스 :  3954.916748046875 / r2스코어 :  0.34146599025706104

# [실습] MaxAbsScaler 스켈링하고 돌려보기. 제일 좋음.
# 로스 :  3427.3046875 / r2스코어 :  0.4293187362085088

# [실습] RobustScaler 스켈링하고 돌려보기.
# 로스 :  4364.806640625 / r2스코어 :  0.2732151973621969

# 세이브한 가중치
# 로스 :  3429.9091796875
# r2스코어 :  0.42888511775979776

# 로스 :  3429.9091796875
# r2스코어 :  0.42888511775979776

# 드롭아웃하고
# 로스 :  4094.3388671875 / # r2스코어 :  0.3182507928510956 / # 걸린시간 :  0.87 초

# dnn 데이터 -> cnn데이터로 바꾸기
# 로스 :  6159.8251953125 / r2스코어 :  -0.02567384792998384 / 걸린시간 :  3.07 초
# 로스 :  6197.2431640625 / r2스코어 :  -0.031904342122865526  걸린시간 :  4.0 초

# 모델 Conv1D 
# 로스 :  6193.19921875 / r2스코어 :  -0.031230877528812595 / 걸린시간 :  3.52 초