# keras40_hamsu01_mnist.py 카피

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(60000,28*28)
# x_test = x_test.reshape(10000,28*28)
# print(x_train.shape, x_test.shape)  # (60000, 28*28) (10000, 28*28) 

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    # height_shift_range=0.1, # 평행이동 수직
    # rotation_range= 15,      # 정해진 각도만큼 이미지 회전 
    zoom_range=0.1,         # 축소 또는 확대
    shear_range=0.1,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

augment_size = 40000  # 증가시키다.


randidx = np.random.randint(x_train.shape[0], size=augment_size)   

x_augmented = x_train[randidx].copy()   # .copy()하면 메모리값을 새로 할당하기 때문에 원래 메모리값에 영향을 미치지 않는다. 메모리 안전빵.
y_augmented = y_train[randidx].copy()  

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],          
    x_augmented.shape[1],           
    x_augmented.shape[2], 1)       

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

# print(x_augmented.shape)    # (40000, 28, 28, 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델
model = Sequential()
model.add(Conv2D(128, (3,3), activation='relu', input_shape=(28, 28, 1),
                 strides=2,
                 padding='same'))  
model.add(MaxPooling2D()) 
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
                 strides=1, padding='valid'))   
model.add(MaxPooling2D()) 
model.add(Conv2D(64,(2,2), activation='relu', strides=2, padding='same'))                         
model.add(Flatten())                              
model.add(Dense(units=32, activation='relu')) 
model.add(Dense(units=16, input_shape=(32,)))
                       
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
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
filepath = "".join([path, 'k49_mnist', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( 
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=10000, batch_size= 32,
          validation_split=0.3, verbose=1, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)  

y_pred = model.predict(x_test)
# print(y_pred.shape) # (10000, 10)

# [실습] 아래에 디코딩을 해야됨.

y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
# print(y_pred)
# print(y_pred.shape) # (10000, 1)

y_test = np.argmax(y_test, axis=1).reshape(-1,1)
# print(y_test)
# print(y_test.shape) # (10000, 1)

acc = accuracy_score(y_test, y_pred)

print("로스는 : ", round(loss[0], 3))
print("ACC : ", round(loss[1], 3))
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# ACC 0.98 만들기
# ACC :  0.918
# ACC :  0.92
# ACC :  0.984

# strides=2, padding='same' 넣어서 성능 개선해보기
# ACC :  0.988

# MaxPooling 넣어서 성능 개선해보기
# 로스는 :  0.04743555933237076 / ACC :  0.985 / 걸린시간 :  100.52 초

# 데이터 쫙 피고 다시 돌림.
# 로스는 :  0.18211272358894348 / ACC :  0.958 / 걸린시간 :  83.54 초
# 로스는 :  0.12428952753543854 / ACC :  0.968 / 걸린시간 :  55.55 초
# 로스는 :  0.11565091460943222 / ACC :  0.975 / 걸린시간 :  387.3 초
# 로스는 :  0.12298475205898285 / ACC :  0.975 / 걸린시간 :  746.67 초

# 모델 함수로 돌림.
# 로스는 :  0.09044909477233887 / ACC :  0.9 / 걸린시간 :  3.6 초

# 로스는 :  0.003 / ACC :  0.98 / 걸린시간 :  8.6 초

# augment하고 돌려보기
# 로스는 :  0.003 / ACC :  0.985 / 걸린시간 :  30.07 초 / 에포 100
# 로스는 :  0.002 / ACC :  0.985 / 걸린시간 :  58.87 초 / 에포 1000

# 이미지 확인하고 돌리기
# 로스는 :  0.002 / ACC :  0.986 / 걸린시간 :  50.76 초
# 로스는 :  0.005 / ACC :  0.972 / 걸린시간 :  73.77 초