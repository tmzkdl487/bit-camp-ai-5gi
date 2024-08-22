import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터  
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

## [실습] 잘라라!!!

x_train = x[:13]
y_train = x[:13]

x_val = x[13:15]
y_val = x[13:15]

x_test = x[15:]
y_test = x[15:]

print(x_train, y_train, x_val, y_val, x_test, y_test)

#2. 모델
model = Sequential()
model.add(Dense (1,  input_dim= 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([17])

print("로스 : ", loss)
print('[17]의 예측값 : ', results)




