from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
#     horizontal_flip=True,   # 증폭하면 이미지 망가질 수 있어서 주석처리 하라고 하심.
#     vertical_flip=True,    
#     width_shift_range=0.1, 
#     height_shift_range=0.1, 
#     rotation_range= 5,   
#     zoom_range=1.2,        
#     shear_range=0.7,       
#     fill_mode='nearest',   
)

test_datagen = ImageDataGenerator(
    rescale=1./255)

path_train = './_data/image/horse_human/'

start_time1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(100, 100), 
    batch_size=1027, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)  # Found 1027 images belonging to 2 classes.

# x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, shuffle=True, random_state=666)


np_path = 'c:/ai5/_data/_save_npy/'
np.save(np_path + 'keras45_horse_02_x_train.npy' , arr=xy_train[0][0])
np.save(np_path + 'keras45_horse_02_y_train.npy' , arr=xy_train[0][1])

end_time1 = time.time()