# keras28_1_save_model.py 복사

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import time

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=6666)

scaler = RobustScaler() # MinMaxScaler # StandardScale, MaxAbsScaler

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=13))    
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# es = EarlyStopping(
#     monitor= 'val_loss',
#     mode = 'min',
#     patience= 10,
#     verbose= 1,
#     restore_best_weights= True )

# mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_olny=True, 
#     filepath = './_save/keras29_mcp/keras29_mcp1.hdf5'
# )

# start_time = time.time()

# model.fit(x_train, y_train, epochs=1000, batch_size=16,   # hist는 히스토리를 줄인말이다.
#           verbose=1, callbacks = [es, mcp],
#           )
# end_time = time.time()

model = load_model('./_save/keras29_mcp/keras29_mcp3.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("로스 : ", loss)
    
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2) 
# print("걸린시간 : ", round(end_time - start_time, 2), "초")

# [과제] train_size = 0.7 ~ 0.9 사이 / r2 0.8 -0.1 줄여주심. 0.7 나오게 하기.

# train_size=0.75 r2스코어 :  0.7365823081353948
# train_size=0.7 / random_state=15645 / epochs=10000 / random_state=15645 / r2스코어 :  0.7007794834273509
# train_size=0.75 / random_state=8989 / epochs=1000 / batch_size=32 / r2스코어 :  0.7042565389216131
# train_size=0.7 / random_state=333333 /  epochs=3333/ batch_size=33 / r2스코어 :  0.7540095067146827/ r2스코어 :  0.760713613142604 / r2스코어 :  0.7463206127196489 / r2스코어 :  0.7573002851790058
# train_size=0.8 / random_state=333333 / epochs=3333 / batch_size=33 / r2스코어 :  0.779991349020035 / 0.7462829323349436 / r2스코어 :  0.7785035118528679 / r2스코어 :  0.7736757755732411
# r2스코어 :  0.741570838347922 / 0.7721948798609052 / r2스코어 :  0.77252773831965 / 0.7748238761491439
# r2스코어 :  0.7162032498512081 / r2스코어 :  0.7415556719594341 / r2스코어 :  0.7895978890121733
# r2스코어 :  0.789752025067346

# verbose=0, validation_split=0.3을 넣어서 다시 해봄.
# r2스코어 :  0.8318105597373147 / 
# validation_split=0.4을 넣어서 다시 해봄.
# r2스코어 :  0.8390326228175955 / r2스코어 :  0.856450444072601
# 로스 :  18.055999755859375

# [실습] 데이터 MinMaxScaler 스켈링하고 돌려보기.
# 로스 :  20.564085006713867 / r2스코어 :  0.8105601530728421

# [실습] 데이터 StandardScaler 스켈링하고 돌려보기.
# 로스 :  21.749914169311523 / r2스코어 :  0.799636098308407

# [실습] MaxAbsScaler 스켈링하고 돌려보기.
# 로스 :  17.427099227905273 / r2스코어 :  0.8394585791665131

# [실습] RobustScaler 스켈링하고 돌려보기. 제일 좋음.
# 로스 :  16.715578079223633 / r2스코어 :  0.8460132615838727