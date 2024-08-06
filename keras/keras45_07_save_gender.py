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

# #1. 데이터
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True, # 수평 뒤집기
#     vertical_flip=True, # 수직 뒤집기
#     width_shift_range=0.1, # 평행이동 수평 이미지 전체를 10프로만큼 이동시켜준다.
#     height_shift_range=0.1, # 평행이동 수직
#     rotation_range=1, #각도 만큼 이미지 회전 / 1
#     zoom_range=0.1, #축소 또는 확대 0.2로 하면 / 0.8 - 1.2 사이로 무작위
#     shear_range=0.1, # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
#     fill_mode='nearest' #이동했을 때 비는 공간을 가장 가까운 곳의 데이터로 채운다. 예를 들어 주변에 배경이 있다면 그 배경에 가까운 색으로 채워짐
#     )   # Found 27167 images belonging to 1 classes.

# test_datagen = ImageDataGenerator(
#     rescale=1./255)

# path_train = './_data/kaggle/biggest_gender/faces/' 
# path_test = './_data/kaggle/biggest_gender/faces/'


# start_time1 = time.time()

# xy_train = train_datagen.flow_from_directory(
#     path_train, 
#     target_size=(100, 100), 
#     batch_size=27167, 
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True
# ) 

# xy_test = test_datagen.flow_from_directory(
#     path_test, 
#     target_size=(100, 100),  # 타겟 사이즈를 200에 200으로 잡는다.
#     batch_size=160,  # 10, 200, 200, 1로 
#     class_mode='binary',
#     color_mode='grayscale',
#     # shuffle=True, # test 데이터는 shuffle 하지 않음
# )  

# # print(xy_train[0][0].shape) # (27167, 200, 200, 3)
# # print(xy_train[0][1].shape) # (27167,)

# np_path = 'c:/ai5/_data/_save_npy/'
# np.save(np_path + 'keras45_gender_04_x_train.npy' , arr=xy_train[0][0])
# np.save(np_path + 'keras45_gender_04_y_train.npy' , arr=xy_train[0][1])
# np.save(np_path + 'keras45_gender_04_x_test.npy' , arr=xy_test[0][0])
# np.save(np_path + 'keras45_gender_04_y_test.npy' , arr=xy_test[0][1])

############# 여자 데이터만 증폭하기
#1. 데이터
path_train= './_data/kaggle/biggest_gender/faces/'  
path_test = './_data/kaggle/biggest_gender/faces/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 수평 뒤집기
    vertical_flip=True, # 수직 뒤집기
    width_shift_range=0.1, # 평행이동 수평 이미지 전체를 10프로만큼 이동시켜준다.
    height_shift_range=0.1, # 평행이동 수직
    rotation_range=1, #각도 만큼 이미지 회전 / 1
    zoom_range=0.2, #축소 또는 확대 0.2로 하면 / 0.8 - 1.2 사이로 무작위
    shear_range=0.7, # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest' #이동했을 때 비는 공간을 가장 가까운 곳의 데이터로 채운다. 예를 들어 주변에 배경이 있다면 그 배경에 가까운 색으로 채워짐
    )   # Found 27167 images belonging to 1 classes.

test_datagen = ImageDataGenerator(
    rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(80, 80), 
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
) # Found 17678 images belonging to 1 classes.

xy_test = train_datagen.flow_from_directory(
    path_train, 
    target_size=(80, 80), 
    batch_size=20000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
) # Found 9489 images belonging to 1 classes.

np_path = 'c:/ai5/_data/_save_npy/'
# np.save(np_path + 'keras45_gender_05_Man_x_train.npy' , arr=xy_train[0][0][:17678])
# np.save(np_path + 'keras45_gender_05_Man_y_train.npy' , arr=xy_train[0][1][:17678])
# np.save(np_path + 'keras45_gender_woman_05_x_train.npy' , arr=xy_train[0][0][17678:])
# np.save(np_path + 'keras45_gender_woman_05_y_train.npy' , arr=xy_train[0][1][17678:])
np.save(np_path + 'keras45_gender_05_x_test.npy' , arr=xy_test[0][0]) 
np.save(np_path + 'keras45_gender_05_y_test.npy' , arr=xy_test[0][1])

