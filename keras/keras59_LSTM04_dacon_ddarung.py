# keras39_cnn04_dacon_ddarung.py 복사

# https://dacon.io/competitions/open/235576/overview/description (대회 사이트 주소)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import sklearn as sk
import time

#1. 데이터
path = "C://ai5//_data//dacon//따릉이//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 인덱스 없으면 index_col쓰면 안됨. 0은 0번째 줄 없앴다는 뜻이다.

test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)

train_csv = train_csv.dropna()  # 구멍난 데이터를 삭제해달라는 수식

test_csv = test_csv.fillna(test_csv.mean()) # 구멍난 데이터를 평균값으로 채워달라는 뜻.
# print(test_csv.info()) # 715 non-nul /  확인.

x = train_csv.drop(['count'], axis=1)   
y = train_csv['count']  

# print(x.shape)  # (1328, 9)

x = x.to_numpy()
x = x.reshape(1328, 3, 3) 
x = x/255.

# print(x.shape, y.shape) # (1328, 3, 3, 1) (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    random_state=4343,
                                                    shuffle=True,
                                                    ) # random_state=3454, 맛집 레시피 : 4343 / stratify=y

scaler = MinMaxScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#2. 모델 구성
# model = Sequential()
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(3, 3, 1), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Dropout(0.2))
# model.add(Dense(16, input_shape=(32,)))
# model.add(Dense(1))

#2. LSTM모델구성
model = Sequential()
model.add(LSTM(21, return_sequences=True, input_shape=(3, 3), activation='relu')) # timesteps, features
model.add(LSTM(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # metrics=['accuracy', 'acc', 'mse']
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience = 5,
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
filepath = "".join([path, 'k32_dropout_dacon_ddarung', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=1, batch_size=8,
          verbose=1, validation_split=0.2,
          callbacks= [es, mcp]
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

# print(test_csv.shape)   

test_csv = test_csv.to_numpy()
test_csv = test_csv.reshape(715, 3, 3, 1) 

y_submit = model.predict(test_csv)  # 예측한 값을 y_submit에 넣는다는 뜻.
print(y_submit) # 확인 해봄.
print(y_submit.shape)   # (715, 1) / 나왔음.

###################### submissinon.csv만들기 // count컬럼에 값만 넣으주면 돼. ##########

submission_csv['count'] = y_submit  # submission count 열에 y_submit을 넣겠다는 수식.
# print(submission_csv)   # 확인
# print(submission_csv.shape) #.shape확인.

submission_csv.to_csv(path + "submission_0801_1651.csv")    # 폴더 안에 파일로 만들겠다. 가로 안은 (저장 경로 + 파일명)이다.

print("r2스코어 : ", round(r2, 4))
print ("로스는 : ", round(loss[0], 3))  
print("ACC : ", round(loss[1], 3))
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# [실습] MinMaxScaler 스켈링하고 돌려보기.
# r2스코어 :  0.757 / 로스는 :  1349.4285888671875

# [실습] StandardScaler 스켈링하고 돌려보기.
# r2스코어 :  0.6791 / 로스는 :  1781.7999267578125

# [실습] MaxAbsScaler 스켈링하고 돌려보기.
# r2스코어 :  0.7344 / 로스는 :  1474.5343017578125

# [실습] RobustScaler 스켈링하고 돌려보기.
# r2스코어 :  0.7013 / 로스는 :  1658.60986328125

# 세이브한 가중치
# r2스코어 :  0.7396
# 로스는 :  1445.9365234375

# r2스코어 :  0.7396
# 로스는 :  1445.9365234375

# 드롭아웃하고
# r2스코어 :  0.7116 / 로스는 :  1601.3203125

# dnn 데이터 -> cnn데이터로 바꾸기
# 로스는 :  1547.9703369140625 / 걸린시간 :  31.2 초
# 로스는 :  [1623.8275146484375, 0.0] / 걸린시간 :  17.43 초
# r2스코어 :  0.7032 / 로스는 :  1647.666 / 걸린시간 :  14.87 초

# LSTM 모델 
# r2스코어 :  -0.4244 / 로스는 :  7908.249 / ACC :  0.0