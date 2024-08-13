# keras39_cnn02_california.py 복사

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, LSTM  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import sklearn as sk
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape)  # (20640, 8)

x = x.reshape(20640, 4, 2)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=34)

scaler = StandardScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# print(x.shape)  # (20640, 4, 2)

# exit()


#2. 모델 구성
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

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True)

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

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, batch_size=64,
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

# 세이브한 가중치
# # 로스 :  0.2662697434425354
# r2스코어 :  0.8021907608037689

# 로스 :  0.2662697434425354
# r2스코어 :  0.8021907608037689

# 드롭 아웃하고
# r2스코어 :  로스 :  0.41684746742248535/ 0.6903280786715383 / 걸린시간 :  10.0 초

# CPU: 걸린시간 :  2.01 초
# GPU: 걸린시간 :  6.97 초

# dnn 데이터 -> cnn데이터로 바꾸기
# 로스 :  0.45452582836151123 / r2스코어 :  0.6623371485968064 / 걸린시간 :  45.08 초
# 로스 :  0.45029574632644653 / r2스코어 :  0.6654796072854956 / 걸린시간 :  41.86 초

# LSTM 모델 
# 로스 :  1.3275474309921265 / r2스코어 :  0.013777632288543784 / 걸린시간 :  23.47 초