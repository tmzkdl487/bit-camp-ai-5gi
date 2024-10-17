# keras26_scaler01_boston.py 복사

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import sklearn as sk
import time

#1. 데이터
dataset = load_boston()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data
y = dataset.target

# print(x)
# print(x.shape)  # (506, 13)
# print(y)  
# print(y.shape)  # (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=6666)

scaler = RobustScaler() # MinMaxScaler # StandardScale, MaxAbsScaler

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)   # <-요 두 줄을 아래의 1줄로 바꿀 수 있다.
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)  # 소수점으로 나옴.
print(np.min(x_train), np.max(x_train))   # 그래서 다시 찍어봄. 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))   # -0.008298755186722073 1.1478180091225068 <_ 범위 밖으로 나오는게 맞음.

#2. 모델구성
model = Sequential()
# model.add(Dense(1, input_dim=13))
model.add(Dense(10, input_dim=13))    # 이미지 input_shape=(8, 8, 1) / input_shape=(13)
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.save("./_save/keras28/keras28_1_save_model.h5")

'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 0,
    restore_best_weights= True
)

model.fit(x_train, y_train, epochs=10, batch_size=16,   # hist는 히스토리를 줄인말이다.
          verbose=1, validation_split=0.3,
          callbacks = [es]  #얼리스타핑을 콜백한다.
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)
    
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2) 
print("걸린시간 : ", round(end_time - start_time, 2), "초")

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
'''