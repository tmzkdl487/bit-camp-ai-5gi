from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
import pandas as pd
import time

#1. 데이터
dataset = load_wine()

x = dataset.data
y = dataset.target
# print(x.shape, y.shape) # (178, 13) (178,)

y_ohe = pd.get_dummies(y)
print(y_ohe.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9,
                                                    random_state=666,
                                                    stratify=y)

#2. 모델
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=13))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience = 100,
    restore_best_weights= True
)

model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1, validation_split=0.2, callbacks=[es])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print("로스는 : ", loss)
print("ACC : ", round(loss[1], 3))  # 반올림
print("걸린시간 : ", round(end_time - start_time,2), "초")

# ACC :  1.0

# [실습] stratify=y을 넣고 돌려보기.
# ACC :  ACC :  0.389
