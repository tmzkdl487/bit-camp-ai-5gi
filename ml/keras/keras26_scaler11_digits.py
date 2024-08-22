# keras22_softmax4_digits.py 복사

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터

x, y = load_digits(return_X_y=True)  # 데이터 다운하고 x, y 바로 됨.

print(x)
print(y)
print(x.shape, y.shape) # (1797, 64) (1797,)

print(pd.value_counts(y,sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape)  # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9,
                                                    random_state=3434)

# print(x_train.shape, y_train.shape) # (1617, 64) (1617, 10)
# print(x_test.shape, y_test.shape)   # (180, 64) (180, 10)

scaler = RobustScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Dense(256, input_dim=64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu')) 
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 20,
    restore_best_weights= True
)

model.fit(x_train, y_train, epochs= 1000, batch_size=16,
          verbose=1, validation_split=0.2, callbacks=[es])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print("로스는 : ", round(loss[0], 4))
print("ACC : ", round(loss[1], 3))
print("걸린시간: " , round(end_time - start_time, 2), "초")

# [실습] ACC가 1
# ACC :  0.956
# ACC :  0.967
# ACC :  0.972
# ACC :  0.978
# ACC :  0.994
# 로스는 :  0.0366 / AAC :  0.994
# 로스는 :  0.0588 / ACC :  0.994 / patience= 10
# 로스는 :  0.0642 / ACC :  0.994 / patience= 20/ random_state=1186

# [실습2] MinMaxScaler 스켈링
# 로스는 :  0.2776 / ACC :  0.961

# [실습] StandardScaler 스켈링하고
# 로스는 :  0.3064 / ACC :  0.95

# [실습] MaxAbsScaler 스켈링하고 돌려보기.
# 로스는 :  0.3033 / ACC :  0.956

# [실습] RobustScaler 스켈링하고 돌려보기.
# 로스는 :  0.2857 / ACC :  0.95
