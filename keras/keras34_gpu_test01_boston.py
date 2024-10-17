# 01부터 13까지 쭉 카피해서...

# gpu일때, cou일 때의 시간을 

# keras29_ModelCheckPoint1.py 복사

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=6666)

scaler = RobustScaler() # MinMaxScaler # StandardScale, MaxAbsScaler

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_shape=(13,)))    
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

#2-2. 모델 구성 (함수형)
input1 = Input(shape=(13,))
dense1 = Dense(64, name='ys1',  activation='relu')(input1)  # 레이어 이름도 변경가능, 성능에는 영향을 안 미친다.
dense2 = Dense(64, name='ys2',  activation='relu')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(32, name='ys3',  activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(32, name='ys4',  activation='relu')(drop2)
drop3 = Dropout(0.3)(dense4)
dense5 = Dense(16, name='ys5',  activation='relu')(drop3)
drop4 = Dropout(0.2)(dense5)
output1 = Dense(1)(drop4)
model = Model(inputs = input1, outputs = output1)

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

if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxx")

# CPU: 걸린시간 :  2.08 초
# GPU: 걸린시간 :  5.27 초