from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = load_breast_cancer()
# print (datasets)

print(datasets.DESCR)   # DESCR은 교육용만 쓴다.
print(datasets.feature_names)   # 30개의 데이터가 있다.
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

x = datasets.data
y = datasets.target
# [검색] 넘파이 라벨값 찾는 법. (종류 구하는 법, 0과 1의 갯수가 몇개인지 찾아요.)
# [힌트] 넘파이는 유니크 / 판다스는 벨류 카운트

# print(x.shape, y.shape) # (569, 30) (569,) <- 넘파이 데이터
# print(type(x))  # <class 'numpy.ndarray'>라고 나옴.

# print(y.value_counts())  # 에러 남. 판다스에만 됨. 그래서 아래 3개로 알 수 있음.
# print(pd.DataFrame(y). value_counts())  # 판다스에서 카운트 세는 법. print(pd.Series(y). value_counts()) 랑 print(pd.value_counts(y))도 다 똑같다.
#  1    357
#  0    212

# print(pd.Series(y). value_counts()) 
# print(pd.value_counts(y))

# print(np.unique(y, return_counts=True)) 
# (array([0, 1]), array([212, 357], dtype=int64))
# 2진 분류할 때 갯수를 왜 구할까? 불균형 데이터인지 확인하려고.

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8, 
                                                    random_state= 3434)

# print(x_train.shape, y_train.shape) # (455, 30) (455,)
# print(x_test.shape, y_test.shape)   # (114, 30) (114,)

scaler = MaxAbsScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델링
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=30))   # 2의 배수로 많이 함. 64, 32, 16, 8, 2, 1 로 많이 넣음.
model.add(Dense(16, activation = 'relu'))   
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))  # 한정함수가 결과값을 0, 1로 바꾸기 위해서 0, 1 사이의 값으로 한정시킨다. 예시로 37은 1, -0.7은 0이 되는 것이다. 
# [검색] 시그모이드 (sigmoid)검색.
# 시그모이드는 0에서 1사이의 값으로 바꿔주는 것이다.
# 시그모이드를 중간에 넣어도 된다.

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# 시그모이드가 잘되는지 보기 위해서 매트릭스는 훈련에 영향을 미치지 않지만, 보조지표로 보기위해 accuracy를 넣었음.
# 회귀 데이터에는 'accuracy'를 못넣지만 분류에는 'accuracy'를 넣어도 된다. 
# []는 리스트이니 다른 것도 더 넣어도 된다. mse도 넣어도 된다.
# 'accuracy'나 'acc'은 똑같은 말이다.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode ='min',
    patience = 10,
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

path = './_save/keras30_mcp/06_cancer/'
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

model.fit(x_train, y_train, epochs=100, batch_size=32,
verbose=1, validation_split=0.2, callbacks= [es, mcp])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
# print(y_pred[:20])
y_pred = np.round(y_pred)
# print(y_pred[:20])

accuracy_score = accuracy_score(y_test, y_pred)

print("로스 : ", loss[0])
print("ACC : ", round(loss[1], 3))   # 소수 3째자리 만들기
print("acc", accuracy_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# y_predict = model.predict(x_test)
# print(y_predict[:10])

# r2 = r2_score(y_test, y_predict)
# print("r2스코어 : ", r2) 

# 로스는 0에 가깝게, acc는 1과 가깝게 고도화 하기
# 로스 :  0.3333333432674408 / ACC :  0.667
# 로스 :  0.07951367646455765 / ACC :  0.895
# 로스 :  0.06671779602766037 / ACC :  0.921/ acc_score 0.9210526315789473
# 로스 :  0.047147754579782486/ ACC :  0.939 / acc_score 0.9385964912280702

# [실습] MinMaxScaler 스켈링하고 돌려보기.
# 로스 :  0.1481054425239563 / ACC :  0.939

# [실습] StandardScaler 스켈링하고 돌려보기.
# 로스 :  0.0823187381029129 / ACC :  0.965

# [실습] MaxAbsScaler 스켈링하고 돌려보기. 제일 좋음.
# 로스 :  0.05640663206577301 / ACC :  0.974

# [실습] RobustScaler 스켈링하고 돌려보기.
# 로스 :  0.09013770520687103 / ACC :  0.974