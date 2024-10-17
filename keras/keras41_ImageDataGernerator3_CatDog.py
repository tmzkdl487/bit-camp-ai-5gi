# 캣독 맹그러봐!!!

# 1. 에서 데이터에서 시간체크해보기.
    
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

path_train = './_data/image/cat_and_dog/train/'
path_test = './_data/image/cat_and_dog/test/'

start_time1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(100, 100), 
    batch_size=20000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)   # Found 19997 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(100, 100),  
    batch_size=20000,  
    class_mode='binary',
    color_mode='rgb',
    # Found 0 images belonging to 0 classes.
)  

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, 
                                                    shuffle= True,
                                                    random_state=666)

end_time1 = time.time()
# print("데이터 걸린시간 : ", round(end_time1 - start_time1, 2), "초")

# print(xy_train[0][0].shape) # (19997, 80, 80, 3)
# print(xy_train[0][1].shape) # (19997,)

#2. 모델
model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))                        
model.add(Flatten())                  
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])   # acc넣어야 분류일 경우 잘 맞는지 확인할 수 있음.
start_time2 = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights= True
)

########################### mcp 세이프 파일명 만들기 시작 ################
import datetime 
date = datetime.datetime.now()
print(date) # 2024-07-26 16:51:36.578483
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date) # 0726 / 0726_1654
print(type(date))

path = './_save/keras35/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k35_04', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( 
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=1000, batch_size=20,
          validation_split=0.3, verbose=1, callbacks=[es, mcp])

end_time2 = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)  

y_pred = model.predict(x_test)

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("데이터 걸린시간 : ", round(end_time1 - start_time1, 2), "초")
print("걸린시간 : ", round(end_time2 - start_time2, 2), "초")

# 로스는 :  0.5328117609024048 / ACC :  0.869 / 데이터 걸린시간 :  2.31 초 / 걸린시간 :  10.5 초 -> grayscale (흑백으로)
# 로스는 :  0.652917742729187 / ACC :  0.594 / 데이터 걸린시간 :  2.36 초 / 걸린시간 :  15.42 초 -> rgb (컬러로)
# 로스는 :  0.6863765716552734 / ACC :  0.594 / 데이터 걸린시간 :  2.38 초 / 걸린시간 :  13.46 초 -> 잘못함

# 로스는 :  1.790114402770996 / ACC :  0.62 / 데이터 걸린시간 :  7.02 초 / 걸린시간 :  25.81 초
# 로스는 :  0.4768560230731964 / ACC :  0.775 / 데이터 걸린시간 :  39.89 초 / 걸린시간 :  20.94 초 -> batch_size=500
# 로스는 :  0.40897563099861145 / ACC :  0.818 / 데이터 걸린시간 :  39.87 초 / 걸린시간 :  14.3 초

# 로스는 :  0.40575000643730164 / ACC :  0.819 / 데이터 걸린시간 :  39.69 초 / 걸린시간 :  25.77 초 (80* 80)
# 로스는 :  0.41378718614578247/  ACC :  0.805 / 데이터 걸린시간 :  47.83 초 / 걸린시간 :  65.47 초 (100* 100)
# 로스는 :  0.6931115388870239 / ACC :  0.504 / 데이터 걸린시간 :  40.78 초 / 걸린시간 :  119.48 초
