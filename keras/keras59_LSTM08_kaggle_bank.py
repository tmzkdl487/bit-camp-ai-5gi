# keras39_cnn08_kaggle_bank.py 복사

# https://www.kaggle.com/competitions/playground-series-s4e1

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, LSTM  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape)      # (165034, 13)
# print(test_csv.shape)       # (110023, 12)
# print(mission_csv.shape)    # (110023, 1)

# print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

# print(train_csv.isnull().sum())     # 결측치가 없다
# print(test_csv.isnull().sum())

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
# print(x)                            # [165034 rows x 10 columns]
y = train_csv['Exited']
# print(y.shape)                      # (165034,)

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# print(np.unique(y, return_counts=True))     
# # (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))
# print(pd.DataFrame(y).value_counts())
# # 0         130113
# # 1          34921

# print(x.shape)  # (165034, 10)

x = x.to_numpy()
x = x.reshape(165034, 5, 2)
x = x/255.

scaler = MinMaxScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)   # 현아님이 알려준 리쉐입 2번 하는 방법

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    shuffle= True,
                                                    random_state=666)

# # 2. 모델구성
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

#2. LSTM모델구성
model = Sequential()
model.add(LSTM(21, return_sequences=True, input_shape=(5, 2), activation='relu')) # timesteps, features
model.add(LSTM(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

es = EarlyStopping(             # arlyStopping 정의
    monitor='val_loss', 
    mode = 'min',               # 모르면 auto
    patience=10,
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

model.fit(x_train, y_train, epochs=100, batch_size=512, 
                 validation_split=0.2, 
                 callbacks=[es, mcp])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
# print(test_csv.shape)   # (110023, 10)

y_predict = np.round(y_predict)
# print(y_predict[:20])       # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)

test_csv = test_csv.to_numpy()
test_csv = test_csv.reshape(110023, 5, 2, 1)

y_submit = model.predict(test_csv)
# print(y_submit.shape)       # (110023, 1)

y_submit = np.round(y_submit)
mission_csv['Exited'] = y_submit
mission_csv.to_csv(path + "sample_submission_keras59_LSTM08_kaggle_bank_0813_2008.csv")

print("로스는 : ", round(loss[0], 3))
print("ACC : ", round(loss[1], 3))
print("걸린시간 : ", round(end_time - start_time, 2), "초")


# if(gpus):
#     print("쥐피유 돈다!!!")
# else:
#     print("쥐피유 없다! xxxx")

# CPU: 걸린시간 :  10.92 초
# GPU: 걸린시간 :  52.76 초

# dnn 데이터 -> cnn데이터로 바꾸기
# loss :  3.2890305519104004 / accuracy :  0.787 / 걸린시간 :  13.46 초

# LSTM 모델 
# 로스는 :  0.448 / ACC :  0.79 / 걸린시간 :  27.63 초  / 점수 0.54831 <- 10 에포 
# 로스는 :  0.492 / ACC :  0.787 / 걸린시간 :  41.45 초 / 점수 0.52345 <- 100 에포
# 로스는 :  0.436 / ACC :  0.801 / 걸린시간 :  141.74 초 / 점ㅈ수 0.53247 <- 100에포2