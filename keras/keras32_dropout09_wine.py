# keras30_MCP_save_09_wine.py 복사

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
print(y_ohe.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9,
                                                    random_state=666,
                                                    stratify=y)
scaler = RobustScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=13))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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

path = './_save/keras32'
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

print("로스 : ", loss)
print("ACC : ", round(loss[1], 3))  # 반올림
print("걸린시간 : ", round(end_time - start_time,2), "초")

# ACC :  1.0

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
