from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#[실습] 레이어의 깊이와 노드의 갯수를 이용해서 [6]을 맹그러
# 에포는 100으로 고정, 건들이기 말 것!!!
# 소수 네자리까지 맞추면 합격. 예: 6.0000 또는 5.9999

#2. 모델
model = Sequential()
model.add(Dense(33, input_dim=1))
model.add(Dense(3330, input_dim=33))
model.add(Dense(5000, input_dim=3330))
model.add(Dense(33, input_dim=5000))
model.add(Dense(500, input_dim=33))
model.add(Dense(5000, input_dim=500))
model.add(Dense(30, input_dim=5000))
model.add(Dense(500, input_dim=30))
model.add(Dense(30, input_dim=500))
model.add(Dense(500, input_dim=30))
model.add(Dense(50, input_dim=500))
model.add(Dense(30, input_dim=50))
model.add(Dense(3, input_dim=30))
model.add(Dense(1, input_dim=3))

epochs = 100
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측
loss = model.evaluate(x,y)  # 평가하는 것.
print("==========================")
print("epochs : ", epochs)
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 : ", result)

# 소수 네자리까지 맞추면 합격. 예: 6.0000 또는 5.9999

# 6의 예측값 :  [[5.8433704]]
# 6의 예측값 :  [[3.5970716]]
# 6의 예측값 :  [[5.6891384]]
# 6의 예측값 :  [[5.9297557]]
# 6의 예측값 :  [[5.981786]]
# 6의 예측값 :  [[5.985612]]
# 6의 예측값 :  [[6.1875005]]
# 6의 예측값 :  [[5.6127253]]
# 6의 예측값 :  [[6.334153]]
# 6의 예측값 :  [[6.318957]]
# 6의 예측값 :  [[72.63531]]
# 6의 예측값 :  [[6.5337806]]
# 6의 예측값 :  [[83.79966]]
# 6의 예측값 :  [[5.9661174]]
# 6의 예측값 :  [[5.985497]]
# 6의 예측값 :  [[6.0201273]]
# 6의 예측값 :  [[5.9418507]]
# 6의 예측값 :  [[6.011881]]
# 6의 예측값 :  [[5.902652]]
# 6의 예측값 :  [[6.1104016]]
# 6의 예측값 :  [[5.466174]]
# 6의 예측값 :  [[5.8804526]]
# 6의 예측값 :  [[6.356597]]
# 6의 예측값 :  [[6.071876]]
# 6의 예측값 :  [[5.9407697]]
# 6의 예측값 :  [[6.060328]]
# 6의 예측값 :  [[5.9617586]]
# 6의 예측값 :  [[6.057018]]
# 6의 예측값 :  [[5.9929695]]
# 6의 예측값 :  [[5.981186]]
# 6의 예측값 :  [[5.7989745]]
# 6의 예측값 :  [[6.108038]]
# 6의 예측값 :  [[5.910901]]
# 6의 예측값 :  [[6.036428]]
# 6의 예측값 :  [[5.9514694]]
# 6의 예측값 :  [[6.0084643]]