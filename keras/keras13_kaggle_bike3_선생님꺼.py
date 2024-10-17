# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv (카글 컴피티션 사이트)

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = 'C:\\ai5\_data\\bike-sharing-demand\\'   # 절대 경로. 폴더 상단을 누르면 경로를 복사할 수 있다. \a \b는 특수문자라서 슬러시 한개를 더친다.

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test2_csv = pd.read_csv(path + "test2.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)  # (10886, 11)
print(test2_csv.shape)   # (6493, 8)
print(sampleSubmission.shape)   # (6493, 1)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],

print(train_csv.info()) # non-null확인 결측치 확인. 
print(test2_csv.info())  # non-null확인 null이 없다.

print(train_csv.describe()) # mean 평균값, std, min은 최소값, 
# 위에 수식을 돌리면 이렇게 나옴.
# count
# mean -> 평균값
# std -> 표준편차
# min -> 최소값
# 25% -> 1/4 분위
# 50% -> 중위값 3/4 분위값
# 75% -> 분위값
# max

########## 결측치 확인 ################
print(train_csv.isna().sum())   # 없다고 나옴.
print(train_csv.isnull().sum()) # 없다고 나옴.
print(test2_csv.isna().sum())    # 없다고 나옴.    
print(test2_csv.isnull().sum())  # 없다고 나옴.

########### x와 y를 분리
x  = train_csv.drop(['count'], axis=1)   # test는 casual, registered를 더하면 count이라고 함.
# 대가로는 1개. 2개 이상은 리스트
print(x)

y = train_csv['count']
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=33)

#2. 모델 구성   모래시계 모형은 안됨.
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=10))
model.add(Dense(123, activation='relu'))
model.add(Dense(113, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(93, activation='relu'))
model.add(Dense(83, activation='relu'))
model.add(Dense(73, activation='relu'))
model.add(Dense(63, activation='relu'))
model.add(Dense(53, activation='relu'))
model.add(Dense(43, activation='relu'))
model.add(Dense(33, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear')) 

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))
          
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

y_submit = model.predict(test2_csv)
print(y_submit)
print(y_submit.shape)   # (10886,)

sampleSubmission['count'] = y_submit
print(sampleSubmission)
print(sampleSubmission.shape)   #(10886,)

sampleSubmission.to_csv(path + "sampleSubmission_0718_1420.csv")

print ("로스는 : ", loss)
