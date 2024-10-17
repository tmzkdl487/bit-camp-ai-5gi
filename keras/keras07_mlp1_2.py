import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터  # 메트릭트 해봄.
x = np.array([[1,2,3,4,5],
              [1.1,1.2,1.3,1.4,1.5]])
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

x = x.T
# x = x.transpose()
# x= np.transpose(x)

print(x.shape) # (5, 2)
print(y.shape) # (5,)

"""
#2. 모델
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
