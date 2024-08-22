from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터  # X가 3개 y도 여러개.
x = np.array([range(10), range(21, 31), range(201, 211)])

y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [10,9,8,7,6,5,4,3,2,1]])

print(x.shape)
print(y.shape)

x = x.T
y = np.transpose(y)
print(x.shape)  # (10, 3)
print(y.shape)  # (10, 2)

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print('로스 : ', loss)
print('[10, 31, 211]의 예측값 : ', results)

np.set_printoptions(precision=3, suppress=True) 
# 숫자를 예쁘게 보여주는 방법을 설정 / 
# precision=3: 숫자를 소수점 아래 3자리까지만 보여주는 것
# suppress=True: 숫자를 "e"(지수 표기법) 없이 그냥 보통 숫자처럼 보여주게 하는 거야.

#[실습] 맹그러봐
# X_predict = [10, 31, 211] 11과 0이 나오게.

# epochs=2000, batch_size=3 로스 :  0.04710306599736214 [10, 31, 211]의 예측값 :  [[11.1120205   0.35846326]]
# epochs=2000, batch_size=1 로스 :  0.002043291227892041 [10, 31, 211]의 예측값 :  [[10.955939   -0.04910766]]
# epochs=1500, batch_size=1 로스 :  0.6077855229377747 [10, 31, 211]의 예측값 :  [[11.614177   1.0798676]