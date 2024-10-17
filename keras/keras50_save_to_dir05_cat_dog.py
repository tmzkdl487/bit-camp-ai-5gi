# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview

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

# print(x_train.shape, y_train.shape) # (44997, 80, 80, 3) (44997,)
# print(x_test.shape, y_test.shape) # (12500, 80, 80, 3) (12500,)

augment_size = 5000

randidx = np.random.randint(x_train.shape[0], size=augment_size)    

x_augmented = x_train[randidx].copy()   # .copy()하면 메모리값을 새로 할당하기 때문에 원래 메모리값에 영향을 미치지 않는다. 메모리 안전빵.
y_augmented = y_train[randidx].copy()   #  x, y 5만개 준비됨.

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range= 2,      # 정해진 각도만큼 이미지 회전 
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

# x_augmented = x_augmented.reshape(
#     x_augmented.shape[0],         
#     x_augmented.shape[1],          
#     x_augmented.shape[2], 3) 

# print(x_augmented.shape)    # (5000, 102, 102, 3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='c:/ai5/_data/_save_img/05_cat_dog/'
).next()[0]