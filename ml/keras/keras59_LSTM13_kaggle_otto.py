# keras39_cnn13_kaggle_otto.py 복사

# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

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
path = 'C://ai5//_data//kaggle//otto-group-product-classification-challenge//'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
 
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
    
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# [누리님 조언] 타겟을 숫자로 바꾼다.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
# print(x)    # [61878 rows x 93 columns]

y = train_csv['target']
# print(y.shape)  # (61878,)

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape) 

print(x.shape)  # (61878, 93)

x = x.to_numpy()
x = x.reshape(61878, 31, 3)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.6, shuffle=True, 
                                                    random_state=3, 
                                                    stratify=y)

# print(x_train.shape, y_train.shape) # (46408, 93) (46408,)
# print(x_test.shape, y_test.shape)   # (15470, 93) (15470,)

# scaler = RobustScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = Sequential()
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(31, 3, 1), padding='same'))
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
model.add(LSTM(21, return_sequences=True, input_shape=(31, 3), activation='relu')) # timesteps, features
model.add(LSTM(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])  

start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 10,
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

model.fit(x_train, y_train, epochs= 1, batch_size=615,
          verbose=1, validation_split=0.2, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

# print(test_csv.shape)   #(144368, 93)

# test_csv = test_csv.to_numpy()
# test_csv = test_csv.reshape(144368, 31, 3, 1)

# y_submit = model.predict(test_csv)

# y_submit = np.round(y_submit)

# sampleSubmission_csv[['Class_1','Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit

# sampleSubmission_csv.to_csv(path + "sampleSubmission_캐글_오또_0801_1753.csv")

print("59_kaggle_otto_로스는 : ", round(loss[0], 4))
print("ACC : ", round(loss[1], 3))
print("걸린시간: " , round(end_time - start_time, 2), "초")
    
# if(gpus):
#     print("쥐피유 돈다!!!")
# else:
#     print("쥐피유 없다! xxxx")

# CPU: 걸린시간:  6.08 초
# GPU: 걸린시간:  3.33 초

# dnn 데이터 -> cnn데이터로 바꾸기
# 로스는 :  0.0988 / ACC :  0.889 / 걸린시간:  14.23 초
# 로스는 :  0.0988 /ACC :  0.889 / 걸린시간:  3.2 초

# LSTM 모델 
# 59_kaggle_otto_로스는 :  0.0988 / ACC :  0.889 / 걸린시간:  4.68 초