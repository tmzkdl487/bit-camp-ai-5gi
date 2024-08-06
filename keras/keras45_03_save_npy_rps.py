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
    rescale=1./255)

path_train = './_data/image/rps/'

start_time1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(80, 80), 
    batch_size=2520, 
    class_mode='categorical', # 다중분류 - 원핫도 되서 나와욤.
    color_mode='rgb',
    shuffle=True,
)  

np_path = 'c:/ai5/_data/_save_npy/'
np.save(np_path + 'keras45_rps_03_x_train.npy' , arr=xy_train[0][0])
np.save(np_path + 'keras45_rps_03_y_train.npy' , arr=xy_train[0][1])

