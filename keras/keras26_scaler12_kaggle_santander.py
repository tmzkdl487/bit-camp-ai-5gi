# keras23_kaggle1_santander_customer.py 복사

# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

# 맹그러!!!
# 다중분류인줄 알았더니 이진분류였다!!!
# 다중분류 다시 찾겠노라!!!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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

#2. 모델
model = Sequential()
model.add(Dense(128, input_dim=200, activation='relu'))
# model.add(Dense(450, activation='relu'))
# model.add(Dense(400))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
# model.add(Dense(250, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))    #'softmax' 'linear'

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])   # 'categorical_crossentropy'

start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 5,
    restore_best_weights= True
)

model.fit(x_train, y_train, epochs= 50, batch_size=1000,
          verbose=1, validation_split=0.2, callbacks=[es])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_predict = model.predict(x_test)

y_submit = model.predict(test_csv)

sample_submission_csv['target'] = y_submit

sample_submission_csv.to_csv(path + "sampleSubmission_0725_1710.csv")

print("로스는 : ", round(loss[0], 4))
print("ACC : ", round(loss[1], 3))
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