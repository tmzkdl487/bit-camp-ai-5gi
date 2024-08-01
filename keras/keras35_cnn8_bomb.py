# keras35_cnn7_cifar100.py 복사

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

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
model.add(Conv2D(100, (3,3), input_shape=(32, 32, 3)))
model.add(Conv2D(50, (3,3)))
model.add(Conv2D(20, (3,3)))
model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(10))   
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

model.fit(x_train, y_train, epochs=10000, batch_size=100,
          validation_split=0.3, verbose=1, callbacks=[es])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)

# y_test = y_test.to_numpy()

# y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)

# acc = accuracy_score(y_test, y_pred)

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린시간: ", round(end_time - start_time, 2), "초")

# ACC :  0.235
# ACC :  로스는 :  3.9026308059692383 / ACC :  0.172 / 걸린시간:  516.04 초 <- 배치 500
# 로스는 :  4.570847034454346 / ACC :  0.108 / 걸린시간:  716.3 초
# 로스는 :  3.3694465160369873 / ACC :  0.205 / 걸린시간:  156.54 초