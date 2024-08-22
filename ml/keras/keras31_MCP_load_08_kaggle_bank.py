# https://www.kaggle.com/competitions/playground-series-s4e1
# keras21_kaggle_bank(누리님)버전 복사

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    shuffle= True,
                                                    random_state=666)

scaler = MinMaxScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

'''
#2. 모델구성
model = Sequential()
model.add(Dense(640,activation='relu', input_dim=10))
model.add(Dense(640, activation='relu'))
model.add(Dense(640, activation='relu'))
model.add(Dense(640, activation='relu'))
model.add(Dense(320, activation='relu'))
model.add(Dense(320, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

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

path = './_save/keras30_mcp/08_kaggle_bank/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k29_', date, '_', filename])
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
'''

#4. 평가, 예측
print("==================== 2. MCP 출력 =========================")
model = load_model('./_save/keras30_mcp/08_kaggle_bank/k29_0726_2122_0018-0.334528.hdf5')
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
# print(y_predict[:20])       # y' 결과
y_predict = np.round(y_predict)
# print(y_predict[:20])       # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
# print("acc score : ", accuracy_score)
# print("time : ", round(end_time - start_time, 2), "초")

y_submit = model.predict(test_csv)
# print(y_submit.shape)       # (110023, 1)

y_submit = np.round(y_submit)
mission_csv['Exited'] = y_submit
mission_csv.to_csv(path + "sample_submission_0725_2011_RobustScaler.csv")

print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

'''
32 16 16 16 16 1
train_size=0.8, random_state=3434 / epochs=100, batch_size=16, validation_split=0.2
acc score :  0.7709923664122137

++++++++++++++++++++++++++++++
random_state=6666
random_state=1866
random_state=1186
'''

# 가중치 세이프
# acc score :  0.8623019359529797
# loss :  0.3248925805091858

# loss :  0.3248925805091858
# accuracy :  0.862

