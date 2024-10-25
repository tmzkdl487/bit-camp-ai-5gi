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
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("로스 : ", loss) 
result = model.predict([1,2,3,4,5,6,7])
print("7의 예측값 : ", result)

# 7의 예측값 :  로스 :  1.6484591469634324e-12 / 7의 예측값 :  [[1.0000019] <- 에포 10,000
# 7의 예측값 :  [[0.99999917]
# 7의 예측값 :  [[1.000001 ]
# 7의 예측값 :  [[1.]
# 7의 예측값 :  [[1.]
# 7의 예측값 :  [[1.0085083]
# 7의 예측값 :  [[1.000017 ]
# 7의 예측값 :  [[1.]
# 7의 예측값 :  [[1.0064318]
# 7의 예측값 :  [[ -1.4683859]