# keras35_cnn6_cifar10.py 복사

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)   
y_test = y_test.reshape(-1,1)  
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

#2. 모델
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3)
                 , strides=2, padding='same'))
model.add(MaxPooling2D()) 
model.add(BatchNormalization()) 
model.add(Dropout(0.3))
model.add(Conv2D(128, (2,2), activation='relu', strides=2, padding='same'))
model.add(MaxPooling2D()) 
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), activation='relu', strides=2, padding='same'))
model.add(BatchNormalization()) 
model.add(Conv2D(128, (3,3), activation='relu', strides=2, padding='same'))
model.add(BatchNormalization()) 
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(128, input_shape=(32,)))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 10, 
    restore_best_weights= True
)

############ 세이프 파일명 만들기 시작 ############
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras35/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5'
filepath = "".join([path, 'k35_06', date, '_', filename])
###### mcp 세이프 파일명 만들기 끗 ###############

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=2000, batch_size=16,
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

# ACC :  0.235
# ACC :  로스는 :  3.9026308059692383 / ACC :  0.172 / 걸린시간:  516.04 초 <- 배치 500
# 로스는 :  4.570847034454346 / ACC :  0.108 / 걸린시간:  716.3 초
# 로스는 :  3.3694465160369873 / ACC :  0.205 / 걸린시간:  156.54 초

# strides=2, padding='same' 넣어서 성능 개선해보기
# 로스는 :  2.990478515625 / ACC :  0.289 / 걸린시간:  83.16 초
# 로스는 :  2.817396402359009 / ACC :  0.296 / 걸린시간:  149.98 초
# 로스는 :  2.835911512374878 / ACC :  0.303 / 걸린시간:  565.27 초 / batch_size=16
# 로스는 :  2.89253830909729 / ACC :  0.299 / 걸린시간:  978.04 초 / batch_size=8

# MaxPooling 넣어서 성능 개선해보기
# 로스는 :  2.5927178859710693 / ACC :  0.339 / 걸린시간:  424.72 초