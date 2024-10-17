# 배치를 160으로 잡고,
# x, y 를 추출해서 모델을 맹그러봐
# acc 0.99 이상

''''
batch_size = 160
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][0]
'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
    rescale=1./255) # 평가 데이터므로 리쉐이프하고 절.대.로 변환하지않고 수치화만 한다.

path_train = './_data/image/brain/train/' # 이미지 데이터의 상위 폴더만 적으면 나머지 폴더 2개의 데이터들은 각각 0, 1로 바뀐다.
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, # 트레인 폴더에 있는 것을 수치화해줘라.
    target_size=(200, 200),  # 타겟 사이즈를 200에 200으로 잡는다.
    batch_size=160,  # 10, 200, 200, 1로 
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)   # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(200, 200),  # 타겟 사이즈를 200에 200으로 잡는다.
    batch_size=160,  # 10, 200, 200, 1로 
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True, # test 데이터는 shuffle 하지 않음
)  

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

#2. 모델
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))                        
model.add(Flatten())                  
model.add(Dense(512, activation='relu'))  
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])   # acc넣어야 분류일 경우 잘 맞는지 확인할 수 있음.
start_time = time.time()

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
filepath = "".join([path, 'k41_brain', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( 
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=100, batch_size=64,
          validation_split=0.3, verbose=1, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)  

y_pred = model.predict(x_test)

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# acc 0.99 이상
# 로스는 :  0.04207681119441986 / ACC :  0.992 / 걸린시간 :  32.71 초