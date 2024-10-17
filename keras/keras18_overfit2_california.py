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

print(x)
print(y)
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=11)

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=8))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=64, 
          verbose=1, validation_split=0.2)
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
plt.figure(figsize=(9, 6))  # 그림판 사이즈
plt.plot(hist.history['loss'], c ='red', label='loss')  #  marker='.'
plt.plot(hist.history['val_loss'], c ='blue', label='val_loss')
plt.legend(loc='upper right')   # 오른쪽 상단에 라벨값 써줌.
plt.title('California Housing')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
               
# [실습] 맹그러 / R2 0.59 이상 -0.1 줄여주심.
# r2스코어 :  0.5343505878314927
# r2스코어 :  0.5153000248745865
# r2스코어 :  0.5407938909590657
# r2스코어 :  0.5532819039210624
# r2스코어 :  0.5520273955003091
# r2스코어 :  0.5884558602863272

#  verbose=0, validation_split=0.2을 넣어봄
# 

