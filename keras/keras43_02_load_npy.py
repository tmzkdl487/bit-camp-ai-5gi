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

start_time1 = time.time()

np_path = 'c:/ai5/_data/_save_npy/'
# np.save(np_path + 'keras43_01_x_train.npy' , arr=xy_train[0][0])
# np.save(np_path + 'keras43_01_x_train.npy' , arr=xy_train[0][1])
# np.save(np_path + 'keras43_01_x_test.npy' , arr=xy_test[0][0])
# np.save(np_path + 'keras43_01_x_test.npy' , arr=xy_test[0][1])

x_train = np.load(np_path + 'keras43_01_x_train.npy')
y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')


print(x_train)
print(x_train.shape)    # 걸린시간 :  1.09 초
print(y_train)
print(y_train.shape)    # 걸린시간 :  1.1 초
print(x_test)
print(x_test.shape)    # 걸린시간 :  1.1 초
print(y_test)
print(y_test.shape)    # 걸린시간 :  1.12 초


end_time1 = time.time()
print("데이터 걸린시간 : ", round(end_time1 - start_time1, 2), "초")    # 데이터 걸린시간 :  0.02 초