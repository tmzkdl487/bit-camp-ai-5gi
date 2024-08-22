# https://dacon.io/competitions/official/236068/overview/description

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
import pandas as pd
import time

#1. 데이터
path = 'C://ai5/_data/dacon/diabetes/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)    # [652 rows x 9 columns]

test_csv = pd.read_csv(path + "test.csv", index_col = 0)
print(test_csv) # [116 rows x 8 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape, test_csv.shape, sample_submission_csv.shape) 
# (652, 9) (116, 8) (116, 1)

print(train_csv.info())

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=133)

print(x_train.shape, y_train.shape) # (65, 8) (65,) 
print(x_test.shape, y_test.shape)   # (587, 8) (587,)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 1,
    restore_best_weights=True,
)
model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1, validation_split=0.3)
end_time = time.time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
print(y_pred[:20])
y_pred = np.round(y_pred)
print(y_pred[:20])

accuracy_score = accuracy_score(y_test, y_pred)

print("acc_score", accuracy_score)
# print("걸린시간: ", round(end_time - start_time, 2), "초")

y_submit = model.predict(test_csv)

y_submit = np.round(y_submit)

sample_submission_csv['Outcome'] = y_submit

sample_submission_csv.to_csv(path + "sample_submission_0723_1145.csv")

# print("ACC : ", round(loss[1],3))
# print("로스 : ", loss[0])