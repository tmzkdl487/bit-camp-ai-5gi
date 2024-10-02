import tensorflow as tf
print(tf.__version__) # 2.16.2 텐서플로우 버전 이였는데 이제 2.7.4버전이다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
print(tf.__version__) #2.16.2

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1)) # 인풋 한 덩어리, 아웃풋 한 덩어리

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #컴퓨터가 알아먹게 컴파일한다.
model.fit(x, y, epochs=10)

#4. 평가, 예측
result = model.predict(np.array([4]))
print("4의 예측값 : ", result)

# 4의 예측값 :  [[1.675651]]
# 4의 예측값 :  [[6.109565]]
# 4의 예측값 :  [[-4.675229]]
# 4의 예측값 :  [[5.405947]]
# 4의 예측값 :  [[-1.656715]]
# 4의 예측값 :  [[5.8807034]]
# 4의 예측값 :  [[-3.5865724]]
# 4의 예측값 :  [[-0.032795]]
# 4의 예측값 :  [[3.9980304]]
# 4의 예측값 :  [[2.6033964]]
# 4의 예측값 :  [[-4.262508]]
# 4의 예측값 :  [[-0.09671324]]
# 4의 예측값 :  [[-4.313154]]
# 4의 예측값 :  [[2.6633625]]
# 4의 예측값 :  [[2.246942]]
# 4의 예측값 :  [[5.1319766]]
# 4의 예측값 :  [[1.8138092]]
# 4의 예측값 :  [[0.30289385]]