from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import sklearn as sk
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=34)

#2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=8))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
start_time = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True,
)

hist = model.fit(x_train, y_train, epochs=10, batch_size=64,
                 verbose=1, validation_split=0.2,
                 callbacks= [es]
                 )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print ("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

print("========================== hist ==============================")
print(hist)
print("======================= hist.histroy =========================")
print(hist.history)
print("============================ loss ============================")
print(hist.history['loss'])
print("======================= val_loss ============================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c = 'red', label='loss')
plt.plot(hist.history['val_loss'], c = 'blue', label='val_loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()