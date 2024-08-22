# keras30_MCP_save_12_kaggle_santander.py 복사

# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
path = 'C://ai5/_data/kaggle/santander-customer-transaction-prediction/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [200000 rows x 201 columns]

test_csv = pd.read_csv(path + "test.csv", index_col = 0)
# print(test_csv) # [200000 rows x 200 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape, test_csv.shape, sample_submission_csv.shape)
# (200000, 201) (200000, 200) (200000, 1)

x  = train_csv.drop(['target'], axis=1) 
# print(x)    #[200000 rows x 200 columns]

y = train_csv['target']
# print(y.shape)  # (200000,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True,
                                                    random_state=3,
                                                    stratify=y)
scaler = RobustScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, y_train.shape) # (100000, 200) (100000,)
# print(x_test.shape, y_test.shape)   # (100000, 200) (100000,)

# #2. 모델
# model = Sequential()
# model.add(Dense(128, input_dim=200, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))    #'softmax' 'linear'

#2-2. 모델 구성 (함수형)
input1 = Input(shape=(200,))
dense1 = Dense(128, name='ys1')(input1)  # 레이어 이름도 변경가능, 성능에는 영향을 안 미친다.
dense2 = Dense(128, name='ys2', activation='relu')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(64, name='ys3', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(32, name='ys4', activation='relu')(drop2)
output1 = Dense(1)(dense4, activation='sigmoid')
model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])   # 'categorical_crossentropy'

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

model.fit(x_train, y_train, epochs= 50, batch_size=1000,
          verbose=1, validation_split=0.2, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_predict = model.predict(x_test)

y_submit = model.predict(test_csv)

sample_submission_csv['target'] = y_submit

sample_submission_csv.to_csv(path + "sampleSubmission_0725_1710.csv")

print("kaggle_santander 로스는 : ", round(loss[0], 4))
print("kaggle_santander ACC : ", round(loss[1], 3))
print("걸린시간: " , round(end_time - start_time, 2), "초")

# ACC :  0.383
# ACC :  0.9

#[실습] MinMaxScaler 스켈링
# 로스는 :  0.2343 / ACC :  0.915

# [실습] StandardScaler 스켈링하고 고도화 1등 점수 0.92573 / 내 최고 점수 : 0.84365
# 로스는 :  0.244 / ACC :  0.9 / 점수 0.76
# 로스는 :  0.241 / ACC :  0.912 / 점수 0.76

# [실습] MaxAbsScaler 스켈링하고 돌려보기.
# 로스는 :  0.2381 / ACC :  0.913

# [실습] RobustScaler 스켈링하고 돌려보기.
# 로스는 :  0.2409 / ACC :  0.91

# 가중치
# 로스는 :  0.2407
# ACC :  0.911

# 로스는 :  0.2407
# ACC :  0.911

# 드롭아웃 하고 나서
# kaggle_santander 로스는 :  0.2398 / kaggle_santander ACC :  0.911 / 걸린시간:  6.4 초