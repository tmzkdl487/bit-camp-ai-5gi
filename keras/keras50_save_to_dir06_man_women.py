# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
np_path = 'c:/ai5/_data/_save_npy/'

x_train_m = np.load(np_path + 'keras45_gender_05_Man_x_train.npy')
y_train_m = np.load(np_path + 'keras45_gender_05_Man_y_train.npy') 
x_train_w = np.load(np_path + 'keras45_gender_woman_05_x_train.npy')  # 
y_train_w = np.load(np_path + 'keras45_gender_woman_05_y_train.npy')  #  
x_test = np.load(np_path + 'keras45_gender_05_x_test.npy')  
y_test = np.load(np_path + 'keras45_gender_05_y_test.npy')  

# print(x_train_w.shape, y_train_w.shape) # (9489, 80, 80, 3) (9489,)
# print(x_train_m.shape, y_train_m.shape) # (17678, 80, 80, 3) (17678,)
# print(x_test.shape, y_test.shape) # (20000, 80, 80, 3) (20000,)

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.2,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range= 15,      # 정해진 각도만큼 이미지 회전 
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

augment_size = 9000

randidx = np.random.randint(x_train_w.shape[0], size=augment_size)    

x_augmented = x_train_w[randidx].copy()   # .copy()하면 메모리값을 새로 할당하기 때문에 원래 메모리값에 영향을 미치지 않는다. 메모리 안전빵.
y_augmented = y_train_w[randidx].copy()

# print(x_augmented.shape)    # (9000, 80, 80, 3)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],         
    x_augmented.shape[1],          
    x_augmented.shape[2], 3) 

# print(x_augmented.shape)    # (9000, 80, 80, 3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='c:/ai5/_data/_save_img/06_man_women/'
).next()[0]