# keras39_cnn12_kaaggle_santander.py 복사

# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, LSTM
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

# print(x.shape)  # (200000, 200)

x = x.to_numpy()
x = x.reshape(200000, 100, 2)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True,
                                                    random_state=3,
                                                    stratify=y)
scaler = RobustScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train.shape, y_train.shape) # (100000, 200) (100000,)
# print(x_test.shape, y_test.shape)   # (100000, 200) (100000,)

# 2. 모델
# model = Sequential()
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(10, 10, 2), padding='same'))
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
model.add(LSTM(21, return_sequences=True, input_shape=(100, 2), activation='relu')) # timesteps, features
model.add(LSTM(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])   # 'categorical_crossentropy'

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

model.fit(x_train, y_train, epochs= 10, batch_size=1000,
          verbose=1, validation_split=0.2, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_predict = model.predict(x_test)

# print(test_csv.shape)   # (200000, 200)

test_csv = test_csv.to_numpy()
test_csv = test_csv.reshape(200000, 10, 10, 2)

y_submit = model.predict(test_csv)

y_submit = np.round(y_submit)
sample_submission_csv['target'] = y_submit

sample_submission_csv.to_csv(path + "sampleSubmission_keras59_LSTM12_kaaggle_santander_0813_2016.csv")

print("kaggle_santander 로스는 : ", round(loss[0], 4))
print("kaggle_santander ACC : ", round(loss[1], 3))
print("걸린시간: " , round(end_time - start_time, 2), "초")

# if(gpus):
#     print("쥐피유 돈다!!!")
# else:
#     print("쥐피유 없다! xxxx")

# CPU: 걸린시간:  2.6 초
# GPU: 걸린시간:  4.51 초

# dnn 데이터 -> cnn데이터로 바꾸기
# kaggle_santander 로스는 :  0.0727 / kaggle_santander ACC :  0.908 / 걸린시간:  58.72 초

# LSTM 모델 
# kaggle_santander 로스는 :  0.0872 / kaggle_santander ACC :  0.9  / 걸린시간:  133.35 초

# 

