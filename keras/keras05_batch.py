from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# [실습] keras04 의 가장 좋은 레이어와 노드를 이용하여,
# 최소의 loss를 맹그러
# batch_size 조절
# 에포는 100으로 고정을 풀어주겠노라!!!
# 로스 기준 0.31 미만!!!

# print(x.shape, y.shape) # (6,) (6,)

# exit()

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(33))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

epochs = 101
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("==========================")
print("epochs : ", epochs)
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 : ", result)

# 로스 :  0.34441685676574707
# 로스 :  0.3301039934158325
# 로스 :  0.32917603850364685
# 로스 :  0.32725775241851807
# 로스 :  0.32432058453559875
# 로스 :  0.3241700828075409
# 로스 :  0.3241569697856903
# 로스 :  0.3239634931087494
# 로스 :  0.32394149899482727
# 로스 :  0.3239200711250305
# 로스 :  0.32389792799949646
# 로스 :  0.32381391525268555
# 로스 :  0.32381391525268555
