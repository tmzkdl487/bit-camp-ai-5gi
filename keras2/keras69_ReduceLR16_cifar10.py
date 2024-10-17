# keras68_optimizer16_cifar10.py

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

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
(x_train4, y_train2), (x_test, y_test) = cifar10.load_data()

x_train3 = x_train4/255.
x_test = x_test/255.

# print(x_train3.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

# exit()

x_train2 = x_train3.reshape(x_train3.shape[0], x_train3.shape[1]*x_train3.shape[2]*x_train3.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

# print(x_train2.shape, x_test.shape)  # (50000, 3072) (10000, 3072)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x_train2, y_train2, train_size=0.75, 
                                                     shuffle=True, 
                                                     random_state=337)

# print(x_train.shape, y_train.shape) # (37500, 3072) (37500, 1)
# print(x_test.shape, y_test.shape)   # (12500, 3072) (12500, 1)

# exit()

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

# 결과 저장
results = []

for learning_rate in lr:

    #2. 모델
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=x_train.shape[1]))   # relu는 음수는 무조껀 0으로 만들어 준다.
    model.add(Dense(10))
    model.add(Dense(1)) # , activation='softmax'
    
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
# lr: 0.1, r2: -0.00029805464163557005
# ======================= 1. 기본출력 =============================
# lr: 0.01, 로스:0.01
# lr: 0.01, r2: -0.0005701095089414299
# ======================= 1. 기본출력 =============================
# lr: 0.005, 로스:0.005
# lr: 0.005, r2: -0.00015477788475326548
# ======================= 1. 기본출력 =============================
# lr: 0.001, 로스:0.001
# lr: 0.001, r2: 0.022342839170689466
# ======================= 1. 기본출력 =============================
# lr: 0.0005, 로스:0.0005
# lr: 0.0005, r2: 0.00022993459369669011
# ======================= 1. 기본출력 =============================
# lr: 0.0001, 로스:0.0001
# lr: 0.0001, r2: -0.04938168507137819