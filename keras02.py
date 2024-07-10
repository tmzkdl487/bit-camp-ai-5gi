<<<<<<< HEAD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x, y, epochs=1234)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("로스 : ", loss) # 로스를 추가했다.
result = model.predict([1,2,3,4,5,6,7]) #숫자를 추가했다.
=======
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x, y, epochs=10000)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("로스 : ", loss) 
result = model.predict([1,2,3,4,5,6,7])
>>>>>>> 5c45cd04e4a554d256b09cde94621bea124a3fb9
print("7의 예측값 : ", result)