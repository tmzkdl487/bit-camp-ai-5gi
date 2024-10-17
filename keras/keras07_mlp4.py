from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터  # y가 3개인 데이터 
x = np.array([range(10)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
            [10,9,8,7,6,5,4,3,2,1],
             [9,8,7,6,5,4,3,2,1,0]]) 
print(x.shape)  # (1, 10)
print(y.shape)  # (3, 10)

x = x.T
y = y.T
print(x.shape)
print(y.shape)

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(3))
model.add(Dense(33))
model.add(Dense(33))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10]])
print('로스 : ', loss)
print('[10, 0, -1]의 예측값 : ', results)

# 11, 0, -1 이 나와야 됨

# [10, 0, -1]의 예측값 :  [[ 1.1000451e+01  4.2950809e-03 -9.9598122e-01]]