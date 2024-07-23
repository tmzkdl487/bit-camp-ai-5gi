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
geography_mapping = {'Spain': 1, 'France': 2, 'Germany': 3}
gender_mapping = {'Female': 1, 'Male': 2}

train_csv['Geography'] = train_csv['Geography'].map(geography_mapping)
train_csv['Gender'] = train_csv['Gender'].map(gender_mapping)

test_csv['Geography'] = test_csv['Geography'].map(geography_mapping)
test_csv['Gender'] = test_csv['Gender'].map(gender_mapping)

print(train_csv.dtypes)
print(test_csv.dtypes)

train_csv = train_csv.dropna() 
test_csv = test_csv.fillna(test_csv.mean()) 

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

print(x)

# test_csv = test_csv.drop(['CustomerId', 'Surname', 'Balance', 'EstimatedSalary'], axis=1)

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)  

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 25,
    restore_best_weights=True,
)
model.fit(x_train, y_train, epochs=1000, batch_size=512,
          verbose=1, validation_split=0.2, callbacks= [es])
end_time = time.time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
# print(y_pred[:20])
y_pred = np.round(y_pred)
print(y_pred[:50])

accuracy_score = accuracy_score(y_test, y_pred)

y_submit = model.predict(test_csv)

y_submit = np.round(y_submit)

sample_submission_csv['Exited'] = y_submit

sample_submission_csv.to_csv(path + "sample_submission_0723_1216.csv")

print("ACC : ", round(loss[1],3))
print("로스 : ", loss[0])
