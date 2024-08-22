# 01부터 13까지 쭉 카피해서...

# gpu일때, cou일 때의 시간을 

# keras29_ModelCheckPoint1.py 복사

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten,MaxPooling2D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import time

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

# print(x.shape)  # (506, 13)

x = x.reshape(506,13,1,1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=6666)
# print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
# print(y_train.shape, y_test.shape)  # (404,) (102,)

scaler = MinMaxScaler() # MinMaxScaler, StandardScale, MaxAbsScaler, RobustScaler

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(13, 1, 1), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16, input_shape=(32,)))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 10,
    verbose= 1,
    restore_best_weights= True )

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
filepath = "".join([path, 'k32_dropout_bostron', date, '_', filename])
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

hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, 
                 validation_split=0.2, verbose=1, callbacks=[es, mcp])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("로스 : ", loss)
    
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2) 
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 로스 :  13.901333808898926
# r2스코어 :  0.8719385433738247

# 함수형
# 로스 :  11.252641677856445 / r2스코어 :  0.8963387616314299

# dnn 데이터 -> cnn데이터로 바꾸기
# 로스 :  28.6069393157959 / r2스코어 :  0.7364680239259506 / 걸린시간 :  14.79 초