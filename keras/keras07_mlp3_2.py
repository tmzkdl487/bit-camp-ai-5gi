from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
            [10,9,8,7,6,5,4,3,2,1],
             [9,8,7,6,5,4,3,2,1,0]]) 
print(x.shape)
print(y.shape)

x = x.T
y = y.T
print(x.shape)
print(y.shape)

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(3))
model.add(Dense(33))
model.add(Dense(33))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print('로스 : ', loss)
print('[10, 31, 211]의 예측값 : ', results)

# 11, 0, -1 이 나와야 됨

# [10, 31, 211]의 예측값 :  [[4.293675  3.4888358 2.926194 ]]
# [10, 31, 211]의 예측값 :  [[ 1.0995827e+01  1.4795435e-03 -1.0002497e+00]]
# [10, 31, 211]의 예측값 :  [[ 1.1000357e+01 -1.6293675e-04 -1.0000288e+00]]