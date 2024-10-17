# keras68_optimizer13_kaggle_otto.py

# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
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
path = 'C://ai5//_data//kaggle//otto-group-product-classification-challenge//'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [61878 rows x 94 columns]
 
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
# print(test_csv)   # [144368 rows x 93 columns]
    
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(train_csv.shape, test_csv.shape, sampleSubmission_csv.shape)
# (61878, 94) (144368, 93) (144368, 9)

# [누리님 조언] 타겟을 숫자로 바꾼다.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
# print(x)    # [61878 rows x 93 columns]

y = train_csv['target']
# print(y.shape)  # (61878,)

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape) 

# print(x.shape, y.shape) # (61878, 93) (61878,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    random_state=4343,
                                                    shuffle=True,
                                                    )

# print(x_train.shape, y_train.shape) # (455, 13) (455,)
# print(x_test.shape, y_test.shape)   # (51, 13) (51,)

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
# lr: 0.1, r2: -0.980975861994063
# ======================= 1. 기본출력 =============================
# lr: 0.01, 로스:0.01
# lr: 0.01, r2: 0.6609658148733139
# ======================= 1. 기본출력 =============================
# lr: 0.005, 로스:0.005
# lr: 0.005, r2: 0.6670047705934095
# ======================= 1. 기본출력 =============================
# lr: 0.001, 로스:0.001
# lr: 0.001, r2: 0.542667243017157
# ======================= 1. 기본출력 =============================
# lr: 0.0005, 로스:0.0005
# lr: 0.0005, r2: 0.44035996993728954
# ======================= 1. 기본출력 =============================
# lr: 0.0001, 로스:0.0001
# lr: 0.0001, r2: -0.022872841803718735

