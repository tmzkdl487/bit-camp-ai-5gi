# keras68_optimizer10_fetch_covtype.py 복사

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam  

import numpy as np
import pandas as pd
import time

import random as rn
import tensorflow as tf
tf.random.set_seed(337)
np.random.seed(337)
rn.seed(337)

#1. 데이터 
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (581012, 54) (581012,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, 
                                                     shuffle=True, 
                                                     random_state=337)

# print(x_train.shape, y_train.shape) # (435759, 54) (435759,)
# print(x_test.shape, y_test.shape)   # (145253, 54) (145253,)

# exit()

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001] 


# 결과 저장
results = []

for learning_rate in lr:

    #2. 모델
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=x_train.shape[1]))   # relu는 음수는 무조껀 0으로 만들어 준다.
    model.add(Dense(10))
    model.add(Dense(1))
    
    #3. 컴파일, 훈련
    es = EarlyStopping (monitor='val_loss', mode='min',
                        patience=30, verbose=1,
                        restore_best_weights=True,)

    rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                            patience=15, verbose=1, 
                            factor=0.9)
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1,
          batch_size=32, 
          verbose=0,
          callbacks=[es, rlr]
          )

    #4. 평가, 예측
    print("======================= 1. 기본출력 =============================")

    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr: {0}, 로스:{0}'.format(learning_rate, loss))

    y_predict = model.predict(x_test, verbose=0)

    r2 = r2_score(y_test, y_predict)
    print('lr: {0}, r2: {1}'.format(learning_rate, r2))

# ======================= 1. 기본출력 =============================
# lr: 0.1, 로스:0.1
# lr: 0.1, r2: -0.007055019911111016
# ======================= 1. 기본출력 =============================
# lr: 0.01, 로스:0.01
# lr: 0.01, r2: 0.12720757922825499
# ======================= 1. 기본출력 =============================
# lr: 0.005, 로스:0.005
# lr: 0.005, r2: 0.2028332783082445
# ======================= 1. 기본출력 =============================
# lr: 0.001, 로스:0.001
# lr: 0.001, r2: -0.437204663264136
# ======================= 1. 기본출력 =============================
# lr: 0.0005, 로스:0.0005
# lr: 0.0005, r2: -0.7090588008405758

