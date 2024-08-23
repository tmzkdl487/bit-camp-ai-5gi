# keras68_optimizer19_horse.py

# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau    

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
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

path = "./_data/kaggle/dogs-vs-cats-redux-kernels-edition/"
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

start_time1 = time.time()

np_path = 'c:/ai5/_data/_save_npy/'

# 개별 파일 불러오기
x_train_1 = np.load(np_path + 'keras43_01_x_train.npy')
x_train_2 = np.load(np_path + 'keras49_image_cat_dog_01_x_train.npy')
y_train_1 = np.load(np_path + 'keras43_01_y_train.npy')
y_train_2 = np.load(np_path + 'keras49_image_cat_dog_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

# 데이터를 합치기
x_train = np.concatenate((x_train_1, x_train_2), axis=0)
y_train = np.concatenate((y_train_1, y_train_2), axis=0)

xy_test=x_test

x_train = x_train/255.
x_test = x_test/255.

# print(x_train.shape, x_test.shape)  # (49997, 80, 80, 3) (12500, 80, 80, 3)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

# print(x_train.shape, x_test.shape)  # (44997, 19200) (12500, 19200)

# exit()

pca = PCA(n_components=679)  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# print(x_train.shape, x_test.shape)  # (44997, 679) (12500, 679)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.75, 
                                                     shuffle=True, 
                                                     random_state=337)

# print(x_train.shape, x_test.shape)  # (33747, 679) (11250, 679)
# print(y_train.shape, y_test.shape)  # (33747,) (11250,)

# exit()

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(x_train.shape, x_test.shape)  # (33747, 679) (11250, 679)
# print(y_train.shape, y_test.shape)  # (33747, 2) (11250, 2)

# exit()

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

    # print(y_predict.shape)  # (11250, 1)
    
    # print(x_test.shape) # (11250, 679)
    
    # exit()
    
    # y_test_binary = np.argmax(y_test, axis=1)  # 수정: 원-핫 인코딩된 y_test를 이진 레이블로 변환
    # y_predict_binary = np.argmax(y_predict, axis=1) 
    
    r2 = r2_score(y_test, y_predict)
    print('lr: {0}, r2: {1}'.format(learning_rate, r2))
    
# ======================= 1. 기본출력 =============================
# lr: 0.1, 로스:0.1
# lr: 0.1, r2: -0.004064558251237527
# ======================= 1. 기본출력 =============================
# lr: 0.01, 로스:0.01
# lr: 0.01, r2: 0.04838452937471882
# ======================= 1. 기본출력 =============================
# lr: 0.005, 로스:0.005
# lr: 0.005, r2: 0.04930647856558068
# ======================= 1. 기본출력 =============================
# lr: 0.001, 로스:0.001
# lr: 0.001, r2: 0.03501036882058117
# ======================= 1. 기본출력 =============================
# lr: 0.0005, 로스:0.0005
# lr: 0.0005, r2: 0.018846422210592628
# ======================= 1. 기본출력 =============================
# lr: 0.0001, 로스:0.0001
# lr: 0.0001, r2: 0.0041565965464809285   
