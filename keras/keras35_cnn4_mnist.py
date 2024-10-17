# keras35_cnn3_연산량계산_안했다.py 복사

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout  # Flatten = 나라시, 평평하게 하다.
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

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) <- 흑백이미지는 맨 뒤에 1이 생략되어서 오기 때문에 (60000, 28, 28, 1)인 것이다.
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

############ 스케일링 1-1
# x_train = x_train/255.  # x_train은 0 ~ 255 사이로 바뀜
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0

###### 스케일링 1-2
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
print(np.max(x_test), np.min(x_test)) # 1.0 -1.0

# 스케일링 방법 2개인데 둘 다 돌려봐서 더 좋은 것 쓰면 됨.

# ############ 스케일링2 MinMaxSaler(), StandardScaler()
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train))   # 1.0 0.0 위에 리쉐이프 안하면 에러뜸

# ############ 스케일링2 MinMaxSaler(), StandardScaler(), MaxAbsScaler, RobustScaler
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train))   # 244.94693302873063 -1.2742078920822268

############ 스케일링2 MinMaxSaler(), StandardScaler(), MaxAbsScaler, RobustScaler
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler =  MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train))   # 1.0 0.0

# x_train = x_train.reshape(60000, 28, 28, 1) # 스켈링하고 다시 리쉐이프 해줘야됨.
# x_test = x_test.reshape(10000, 28, 28, 1)

############ 원 핫 인코더 1-1 케라스 : to_cateforical (0부터 시작)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train, y_test)
# # [[0. 0. 0. ... 0. 0. 0.]
# #  [1. 0. 0. ... 0. 0. 0.]
# #  [0. 0. 0. ... 0. 0. 0.]
# #  ...
# #  [0. 0. 0. ... 0. 0. 0.]
# #  [0. 0. 0. ... 0. 0. 0.]
# #  [0. 0. 0. ... 0. 1. 0.]] [[0. 0. 0. ... 1. 0. 0.]
# #  [0. 0. 1. ... 0. 0. 0.]
# #  [0. 1. 0. ... 0. 0. 0.]
# #  ...
# #  [0. 0. 0. ... 0. 0. 0.]
# #  [0. 0. 0. ... 0. 0. 0.]
# #  [0. 0. 0. ... 0. 0. 0.]]

# ############ 원 핫 인코더 1-2 판다스 : get_dummies
# y_train = pd.get_dummies(y_train)   # (60000, 28, 28, 1) (60000,)
# y_test = pd.get_dummies(y_test) # (10000, 28, 28, 1) (10000,)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# ############ 원 핫 인코더 1-3 사이킷런 : One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)    # (60000, 28, 28, 1) (60000,)
y_test = y_test.reshape(-1,1)  # (10000, 28, 28, 1) (10000,)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000, 10)

#2. 모델
model = Sequential()
model.add(Conv2D(128, (3,3), activation='relu', input_shape=(28, 28, 1)))   # 26, 26, 64
                        # shape = (batch_size, rows, columns, channels)
                        # shape = (batch_size, height, width, channels)

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))    # 24, 24, 64
model.add(Conv2D(64,(2,2), activation='relu'))                        # 23, 23, 32
model.add(Flatten())                                # 23 * 23 *32

model.add(Dense(units=32, activation='relu'))  #  units=은 이름을 정의해준 것.
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(32,)))
                       
model.add(Dense(10, activation='softmax'))

# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 64)        640

#  conv2d_1 (Conv2D)           (None, 24, 24, 64)        36928

#  conv2d_2 (Conv2D)           (None, 23, 23, 32)        8224

#  flatten (Flatten)           (None, 16928)             0

#  dense (Dense)               (None, 32)                541728

#  dense_1 (Dense)             (None, 16)                528

#  dense_2 (Dense)             (None, 10)                170

# =================================================================
# Total params: 588,218
# Trainable params: 588,218
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])   # acc넣어야 분류일 경우 잘 맞는지 확인할 수 있음.
start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
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

model.fit(x_train, y_train, epochs=150, batch_size= 64,
          validation_split=0.3, verbose=1, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)  

y_pred = model.predict(x_test)
print(y_pred.shape) # (10000, 10)

# [실습] 아래에 디코딩을 해야됨.

y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_pred)
print(y_pred.shape) # (10000, 1)

y_test = np.argmax(y_test, axis=1).reshape(-1,1)
print(y_test)
print(y_test.shape) # (10000, 1)

acc = accuracy_score(y_test, y_pred)

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# ACC 0.98 만들기
# ACC :  0.918
# ACC :  0.92
# ACC :  0.984

# 