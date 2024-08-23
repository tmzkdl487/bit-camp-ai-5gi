# m05_pca_evr_실습19_horse.py

# keras41_ImageDataGernerator4_horse.py 복사

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
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
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    # horizontal_flip=True,   # 증폭하면 이미지 망가질 수 있어서 주석처리 하라고 하심.
    # vertical_flip=True,    
    # width_shift_range=0.1, 
    # height_shift_range=0.1, 
    # rotation_range= 5,   
    # zoom_range=1.2,        
    # shear_range=0.7,       
    # fill_mode='nearest',   
)

test_datagen = ImageDataGenerator(
    rescale=1./255)

path_train = './_data/image/horse_human/'

start_time1 = time.time()

xy_train2 = train_datagen.flow_from_directory(
    path_train, 
    target_size=(10, 10), 
    batch_size=1027, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
) 

# print(xy_train2[0][0].shape, xy_train2[0][1].shape)  # (1027, 10, 10, 3) (1027,)

# exit()

xy_train = xy_train2[0][0].reshape(xy_train2[0][0].shape[0], xy_train2[0][0].shape[1]*xy_train2[0][0].shape[2]*xy_train2[0][0].shape[3])

# print(xy_train.shape) # (1027, 300)

# exit()
x_train, x_test, y_train, y_test = train_test_split(xy_train, xy_train2[0][1], train_size=0.75, 
                                                     shuffle=True, 
                                                     random_state=337)

# print(x_train.shape, y_train.shape) # (770, 300) (770,)
# print(x_test.shape, y_test.shape)   # (257, 300) (257,)

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
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate)) # 'categorical_crossentropy'

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1,
          batch_size=32, 
          verbose=0
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
# lr: 0.1, r2: -0.16381492525051522

# ======================= 1. 기본출력 =============================
# lr: 0.01, 로스:0.01
# lr: 0.01, r2: 0.22427772705173166

# ======================= 1. 기본출력 =============================
# lr: 0.005, 로스:0.005
# lr: 0.005, r2: 0.09443509966998509

# ======================= 1. 기본출력 =============================
# lr: 0.001, 로스:0.001
# lr: 0.001, r2: 0.03006844767425576

# ======================= 1. 기본출력 =============================
# lr: 0.0005, 로스:0.0005
# lr: 0.0005, r2: 0.0052978879478635665

# ======================= 1. 기본출력 =============================
# lr: 0.0001, 로스:0.0001
# lr: 0.0001, r2: -0.28687066688916185