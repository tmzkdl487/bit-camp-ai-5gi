from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#[실습] 레이어의 깊이와 노드의 갯수를 이용해서 최소의 loss을 맹그러
# 에포는 100으로 고정, 건들이기 말 것!!!
# 로스 기준 0.33 이하!!!

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(33))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(33))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(333))
model.add(Dense(333))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(33))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

epochs = 100
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("==========================")
print("epochs : ", epochs)
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 : ", result)

# 로스 기준 0.32 미만!!!

# 로스 :  0.33193376660346985
# 로스 :  0.38058003783226013
# 로스 :  0.3407360017299652
# 로스 :  0.33023712038993835
# 로스 :  0.4089719355106354
# 로스 :  0.36735209822654724
# 로스 :  0.3238096535205841
# 로스 :  0.3260721266269684
# 로스 :  0.32698285579681396
# 로스 :  0.3291389048099518
# 로스 :  0.3463047444820404
# 로스 :  0.35973620414733887
# 로스 :  0.32383444905281067
# 로스 :  0.3295867145061493
# 로스 :  0.865580141544342
# 로스 :  0.38380250334739685
# 로스 :  0.3239360749721527
# 로스 :  0.32388100028038025
# 로스 :  0.32385504245758057
# 로스 :  0.32380911707878113
# 로스 :  0.3569283187389374
# 로스 :  0.3238196074962616
# 로스 :  0.32381725311279297
# 로스 :  0.32469192147254944
# 로스 :  0.3245862126350403
# 로스 :  0.32387590408325195
# 로스 :  0.32386061549186707
# 로스 :  0.3274916112422943
# 로스 :  0.3239351212978363
# 로스 :  0.32638993859291077
# 로스 :  0.32403329014778137
# 로스 :  0.3242007791996002
# 로스 :  0.3243136405944824
# 로스 :  0.32454895973205566
# 로스 :  0.32763203978538513
# 로스 :  0.32389339804649353
# 로스 :  0.32427504658699036
# 로스 :  0.3240720331668854
# 로스 :  0.32415735721588135
# 로스 :  0.3241921663284302
# 로스 :  0.324225515127182
# 로스 :  0.3268636167049408
# 로스 :  0.32487428188323975