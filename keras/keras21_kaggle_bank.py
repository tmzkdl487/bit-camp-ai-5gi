# https://www.kaggle.com/competitions/playground-series-s4e1/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import time

#1. 데이터
path = 'C://ai5/_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + "test.csv", index_col = 0)
# print(test_csv) #[110023 rows x 12 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape, test_csv.shape, sample_submission_csv.shape) 
# (165034, 13) (110023, 12) (110023, 1)

# 'Geography'와 'Gender' 데이터를 숫자로 변환
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=512)

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)  

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 33,
    restore_best_weights=True,
)
model.fit(x_train, y_train, epochs=300, batch_size=64,
          verbose=1, validation_split=0.2)
end_time = time.time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
# print(y_pred[:20])
y_pred = np.round(y_pred)
print(y_pred[:20])

accuracy_score = accuracy_score(y_test, y_pred)

y_submit = model.predict(test_csv)

y_submit = np.round(y_submit)

sample_submission_csv['Exited'] = y_submit

sample_submission_csv.to_csv(path + "sample_submission_0722_1812.csv")

print("ACC : ", round(loss[1],3))
print("로스 : ", loss[0])
