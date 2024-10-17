# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv (카글 컴피티션 사이트)
# 기존 캐글 데이터에서
# 1. train_csv의 y를 casul과 register로 잡는다.
# 그래서 훈련을 해서 test_csv의 casual과 register를 predict한다.
# 2. test_csv에 casual과 register 컬럼을 합쳐
# 3. train_csv에 y를 count로 잡는다.
# 4. 전체 훈련
# 5. test_csv 예측해서 submission에 붙여!

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터 
path = "C:\\ai5\_data\\bike-sharing-demand\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# sampleSubmission_csv = pd.read_csv( path + 'sampleSubmission.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)

y = train_csv[['casual', 'registered']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=11)

#2. 모델 구성 
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=8))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='linear'))  

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

y_submit = model.predict(test_csv)

casual_predict = y_submit[:,0]
registered_predict = y_submit[:,1]

print(casual_predict)

test_csv = test_csv.assign(casual=casual_predict, registered = registered_predict)
test_csv.to_csv(path + "test_columnplus.csv")
