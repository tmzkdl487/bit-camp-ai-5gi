import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  

#1. 데이터 
path = 'C:/ai5/_data/bike-sharing-demand/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test_columnplus.csv", index_col=0)
samplesSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

x = train_csv.drop(['count'], axis=1)

y = train_csv['count']

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=11)

#2. 모델 구성 
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=10))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))  

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=64)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_submit = model.predict(test_csv)
samplesSubmission_csv['count'] = y_submit

samplesSubmission_csv.to_csv(path + "sampleSubmission_col_plus_0718_1229.csv")

print ("로스는 : ", loss)