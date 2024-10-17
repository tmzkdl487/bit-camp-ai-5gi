from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
# x = np.array(range(10))
# print(x)    # [0 1 2 3 4 5 6 7 8 9]
# print(x.shape)  # (10,)

# x = np.array(range(1, 10))  
# print(x)    # [1 2 3 4 5 6 7 8 9]

# x = np.array(range(1, 11))
# print(x)    # [ 1  2  3  4  5  6  7  8  9 10]
# print(x.shape)  # (10,)

x = np.array([range(10), range(21, 31), range(201, 211)])
print(x)
print(x.shape)
x = x.T
print(x)
print(x.shape)  # (10, 3)

y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print('로스 : ', loss)
print('[10, 31, 211]의 예측값 : ', results)

# [실습]
# [10, 31, 211] 예측할 것 = 11이 나와야한다.

# 로스 :  0.0003923896583728492 [10, 31, 211]의 예측값 :  [[10.964533]]
# 로스 :  0.002232056111097336 [10, 31, 211]의 예측값 :  [[10.908955]]
# 로스 :  1.99124503552639e-08 [10, 31, 211]의 예측값 :  [[11.000269]]
# 로스 :  0.14798563718795776 [10, 31, 211]의 예측값 :  [[11.380868]]
# 로스 :  2.999535695380473e-07 / [10, 31, 211]의 예측값 :  [[11.000641]]