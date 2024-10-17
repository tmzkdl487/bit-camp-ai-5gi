# keras22_softmax3_fetch_covtype.py 복사

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터 
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# print(pd.value_counts(y, sort=False))

# 5      9493
# 2    283301
# 1    211840
# 7     20510
# 3     35754
# 6     17367
# 4      2747

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, shuffle=True,
#                                                     random_state=666,
#                                                     stratify=y) #  stratify=y에서 y는 y이다.

# print(pd.value_counts(y_train))
# 2    198085 train_size=0.7
# 1    148291 
# 3     25063 
# 7     14361 
# 6     12277 
# 5      6680 
# 4      1951 

# 2    141627 train_size=0.5
# 1    105772
# 3     17903
# 7     10268
# 6      8774
# 5      4759
# 4      1403

# 2    141650  stratify=y 하니까 라벨을 갯수를 비율에 맞춰서 정확하게 잘라준다.
# 1    105920 
# 3     17877 
# 7     10255 
# 6      8684 
# 5      4746 
# 4      1374 


# print(pd.value_counts(y, ascending=False, sort=False))

# print(x.shape, y.shape) # (581012, 54) (581012,)

# print(y) 

# y_ohe1 = to_categorical(y)  # 케라스, 원핫 인코딩을 1부터 시작하는 데이터이다. 
# print(y_ohe1)   # 위에 숫자가 안나옴.
# print(y_ohe1.shape) # (581012, 8) 

y_ohe = pd.get_dummies(y) # 판다스
# print(y_ohe)    # [581012 rows x 7 columns] / 순서대로 1,2,3,4,5,6,7 으로 나옴.
# # print(y_ohe.shape)  # (581012, 7)

# from sklearn.preprocessing import OneHotEncoder   # 사이킷런
# y_ohe3 = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)    # True가 디폴트
# y_ohe3 = ohe.fit_transform(y_ohe3)
# print(y_ohe3)   # 숫자가 안나옴.
# print(y_ohe3.shape) # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9,
                                                    random_state=666,
                                                    stratify=y)

# print(x_train.shape, y_train.shape) # (522910, 54) (522910, 7)
# print(x_test.shape, y_test.shape)   # (58102, 54) (58102, 7)

scaler = RobustScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=54))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(1024, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 5,
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

path = './_save/keras30_mcp/10_fetch_covtype/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k29_', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs= 100, batch_size=50000,
          verbose=1, validation_split=0.2, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print("로스는 : ", loss)
print("ACC : ", round(loss[1], 3))
print("걸린시간: " , round(end_time - start_time, 2), "초")

# ACC :  1
# ACC :  0.934

# 그냥
# 로스는 :  [0.6457787752151489, 0.7301297783851624] / ACC :  0.73

# [실습] MinMaxScaler스켈링
# 로스는 :  [0.2190803587436676, 0.9135658144950867] / ACC :  0.914

# [실습] StandardScaler 스켈링하고
# 로스는 :  [0.11776526272296906, 0.9570066332817078] / ACC :  0.957

# [실습] MaxAbsScaler 스켈링하고 돌려보기.
# 로스는 :  [0.3821812570095062, 0.8395580053329468] / ACC :  0.84

# [실습] RobustScaler 스켈링하고 돌려보기.
# 로스는 :  [0.24498984217643738, 0.9040480256080627] / ACC :  0.904