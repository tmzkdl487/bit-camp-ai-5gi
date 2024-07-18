import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([[1,2,3,4,5],
#              [1.1,1.2,1.3,1.4,1.5]])
x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

print(x.shape) # (5, 2)
print(y.shape) # (5,)

#2. 모델구성    # 단층 레이어 구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#. 평가, 예측
loss = model.evaluate(x, y)
results = model .predict([[8, 101]])
print('[6, 11]의 예측값 : ', results)

# [실습] : 소수 2째자리까지 맞춰 6이 나와야됨.
# results = model .predict([[6, 11]]) [6, 11]의 예측값 :  [[6.0000005]]
# results = model .predict([[6, 11]]) [6, 11]의 예측값 :  [[5.9926972]
# results = model .predict([[6,11]]) [6, 11]의 예측값 :  [[5.9999433]]]
# results = model .predict([[6,18]]) [6, 11]의 예측값 :  [[5.9999995]]
# results = model .predict([[6,18]]) [6, 11]의 예측값 :  [[5.996835]]
# results = model .predict([[7,12]]) [6, 11]의 예측값 :  [[6.9999933]]
# results = model .predict([[9,15]]) [6, 11]의 예측값 :  [[9.0676565]]
# results = model .predict([[10,11]]) [6, 11]의 예측값 :  [[8.609764]]
# results = model .predict([[8, 101]]) [6, 11]의 예측값 :  [[16.359062]]
