# keras40_hamsu2_fachion.py 복사

from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM   
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

x_train = x_train/255.
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

print(x_train.shape)    # (60000, 28, 28)

# exit()

#2. 모델
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(28*28,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, input_shape=(32,)))
# model.add(Dense(10, activation='softmax'))

#2-2. 모델 구성 (함수형)
# input1 = Input(shape=(28, 28, 1))
# Conv2D1 = Conv2D(128, (3,3), name='ys1',  activation='relu', strides=2, padding='same')(input1)  # 레이어 이름도 변경가능, 성능에는 영향을 안 미친다.
# MaxPooling2D1 = MaxPooling2D()(Conv2D1)
# Conv2D2 = Conv2D(64, (3,3), name='ys2',  activation='relu', strides=2, padding='same')(MaxPooling2D1)
# MaxPooling2D1 = MaxPooling2D()(Conv2D2)
# Conv2D3 = Conv2D(64, (2,2), name='ys3',  activation='relu', strides=2, padding='same')(MaxPooling2D1)
# Flatten1 = Flatten()(Conv2D3)
# Dense1 = Dense(32, activation='relu')(Flatten1)
# Dense2 = Dense(16, activation='relu')(Dense1)
# output1 = Dense(10, activation='softmax')(Dense2)
# model = Model(inputs = input1, outputs = output1)

#2. LSTM모델구성
model = Sequential()
model.add(LSTM(21, return_sequences=True, input_shape=(28, 28), activation='relu')) # timesteps, features
model.add(LSTM(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['acc'])
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 5, 
    restore_best_weights= True
)

############ 세이프 파일명 만들기 시작 ############
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras35/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5'
filepath = "".join([path, 'k35_04', date, '_', filename])
###### mcp 세이프 파일명 만들기 끗 ###############

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=10, batch_size=100,
          validation_split=0.3, verbose=1, callbacks=[es, mcp])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)

# y_test = y_test.to_numpy()

y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test= np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_pred)

print("로스는 : ", round(loss[0], 3))
print("ACC : ", round(loss[1], 3))
print("걸린시간: ", round(end_time - start_time, 2), "초")

# 로스는 :  0.3597969710826874 / ACC :  0.873 / 걸린시간:  6.27 초

# strides=2, padding='same' 넣어서 성능 개선해보기
# 로스는 :  0.2737356126308441 / ACC :  0.902 / 걸린시간:  92.37 초

# MaxPooling 넣어서 성능 개선해보기
# 로스는 :  0.2825157344341278 / ACC :  0.903 / 걸린시간:  109.21 초

# 데이터 쫙 피고 다시 돌림.
# 로스는 :  0.3609013855457306 / ACC :  0.873 / 걸린시간:  502.11 초

# 모델 함수로 돌림.
# 로스는 :  0.09 / ACC :  0.9 / 걸린시간:  15.27 초
# 로스는 :  0.017 / ACC :  0.885 / 걸린시간:  17.07 초

# LSTM 모델 
# 로스는 :  0.09 / ACC :  0.9 / 걸린시간:  130.54 초