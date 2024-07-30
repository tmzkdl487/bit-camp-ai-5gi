from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout  
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

# print(np.unique(y_train, return_counts=True))
# #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
# #       dtype=int64))

# print(pd.value_counts(y_test))
# 9    1000
# 2    1000
# 1    1000
# 6    1000 신발
# 4    1000
# 5    1000
# 7    1000
# 3    1000 원피스
# 8    1000
# 0    1000
# dtype: int64

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train/255.
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1, 1)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.fit_transform(y_test)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


#2. 모델
model = Sequential()
model.add(Conv2D(128, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16, input_shape=(32,)))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 30, 
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

model.fit(x_train, y_train, epochs=1, batch_size=100,
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

# 로스는 :  0.3597969710826874 / ACC :  0.873 / 걸린시간:  6.27 초