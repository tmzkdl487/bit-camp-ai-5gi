# kkeras40_hamsu3_cifar10.py 복사

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = x_train/255.
# x_test = x_test/255.

# x_train = x_train.reshape(50000,32*32*3)
# x_test = x_test.reshape(10000,32*32*3)

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range= 15,      # 정해진 각도만큼 이미지 회전 
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

augment_size = 50000  # 증가시키다.

randidx = np.random.randint(x_train.shape[0], size=augment_size)    

x_augmented = x_train[randidx].copy()   # .copy()하면 메모리값을 새로 할당하기 때문에 원래 메모리값에 영향을 미치지 않는다. 메모리 안전빵.
y_augmented = y_train[randidx].copy()   #  x, y 5만개 준비됨.

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],         
    x_augmented.shape[1],          
    x_augmented.shape[2], 3) 

# print(x_augmented.shape)    # (50000, 32, 32, 3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

# print(x_augmented.shape)    # (50000,32,32,3)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3)

x_train = np.concatenate((x_train, x_augmented))   
y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape) # (100000, 32, 32, 3) (100000, 1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) 
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# #2. 모델
# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(32*32*3,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, input_shape=(32,)))
# model.add(Dense(10, activation='softmax'))

#2-2. 모델 구성 (함수형)
input1 = Input(shape=(32, 32, 3))
Conv2D1 = Conv2D(128, (3,3), name='ys1',  activation='relu', strides=2, padding='same')(input1)  # 레이어 이름도 변경가능, 성능에는 영향을 안 미친다.
MaxPooling2D1 = MaxPooling2D()(Conv2D1)
Conv2D2 = Conv2D(64, (3,3), name='ys2',  activation='relu', strides=2, padding='same')(MaxPooling2D1)
MaxPooling2D1 = MaxPooling2D()(Conv2D2)
Conv2D3 = Conv2D(64, (2,2), name='ys3',  activation='relu', strides=2, padding='same')(MaxPooling2D1)
Flatten1 = Flatten()(Conv2D3)
Dense1 = Dense(32, activation='relu')(Flatten1)
Dense2 = Dense(16, activation='relu')(Dense1)
output1 = Dense(10, activation='softmax')(Dense2)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 20, 
    restore_best_weights= True
)

############ 세이프 파일명 만들기 시작 ############
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras35/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5'
filepath = "".join([path, 'k49_ cifar10', date, '_', filename])
###### mcp 세이프 파일명 만들기 끗 ###############

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=100, batch_size=128,
          validation_split=0.3, verbose=1, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)

# y_test = y_test.to_numpy()

y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_pred)

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린시간: ", round(end_time - start_time, 2), "초")


# [과제] ACC 0.95 만들기
# 로스는 :  1.4067133665084839 / ACC :  0.488 / 걸린시간:  9.68 초
# 로스는 :  1.3710970878601074 / ACC :  0.511 / 걸린시간:  6.73 초
# 로스는 :  1.2581384181976318 / ACC :  0.556 / 걸린시간:  10.56 초 -> batch_size=32
# 로스는 :  1.3472979068756104 / ACC :  0.513 / 걸린시간:  25.12 초 -> batch_size=8
# 로스는 :  1.2817105054855347 / ACC :  0.543 / 걸린시간:  12.26 초 -> batch_size=16
# 로스는 :  1.1666353940963745 / ACC :  0.609 / 걸린시간:  120.88 초
# 로스는 :  1.1442818641662598 / ACC :  0.605 / 걸린시간:  100.18 초
# 로스는 :  1.0635706186294556 / ACC :  0.649 / 걸린시간:  155.34 초

# strides=2, padding='same' 넣어서 성능 개선해보기
# 로스는 :  1.0687144994735718 / ACC : 0.651 / 걸린시간:  247.82 초

# MaxPooling 넣어서 성능 개선해보기
# 로스는 :  0.986098051071167 / ACC :  0.659 / 걸린시간:  254.37 초

# 데이터 쫙 피고 다시 돌림.
# 로스는 :  1.4786052703857422/ ACC :  0.481 / 걸린시간:  76.22 초

# 모델 함수로 돌림.
# 로스는 :  0.09000012278556824 / ACC :  0.9 / 걸린시간:  40.58 초
# 로스는 :  0.04977424442768097 / ACC :  0.637 / 걸린시간:  65.34 초

# augment하고 돌려보기
# 로스는 :  2.3026015758514404 / ACC :  0.1 / 걸린시간:  97.11 초

# 이미지 확인하고 돌리기
# 로스는 :  1.2291247844696045 / ACC :  0.615 / 걸린시간:  114.61 초