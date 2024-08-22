# keras07_mlp2_1.py 복사

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

#1. 데이터  # 컬럼이 3개 짜리
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [9,8,7,6,5,4,3,2,1,0],
             ]
            )
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = x.T

#2. 모델 구성 (순차형)
# model = Sequential()
# model.add(Dense(10, input_shape=(3,)))
# # model.add(Dense(9))
# model.add(Dropout(0.3))
# # model.add(Dense(8))
# model.add(Dropout(0.2))
# # model.add(Dense(7))
# model.add(Dense(1))

# model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 10)                40

#  dense_1 (Dense)             (None, 9)                 99

#  dense_2 (Dense)             (None, 8)                 80

#  dense_3 (Dense)             (None, 7)                 63

#  dense_4 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0
# _________________________________________________________________

#2-2. 모델 구성 (함수형)
input1 = Input(shape=(3,))
dense1 = Dense(10, name='ys1')(input1)  # 레이어 이름도 변경가능, 성능에는 영향을 안 미친다.
dense2 = Dense(9, name='ys2')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(8, name='ys3')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(7, name='ys4')(drop2)
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs = output1)   # 제일 마지막에 어떤 모델인지 정의해줌.

model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 3)]               0

#  dense (Dense)               (None, 10)                40

#  dense_1 (Dense)             (None, 9)                 99

#  dense_2 (Dense)             (None, 8)                 80

#  dense_3 (Dense)             (None, 7)                 63

#  dense_4 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0
# _________________________________________________________________

# 이름 변경 후
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 3)]               0

#  ys1 (Dense)                 (None, 10)                40

#  ys2 (Dense)                 (None, 9)                 99

#  ys3 (Dense)                 (None, 8)                 80

#  ys4 (Dense)                 (None, 7)                 63

#  dense (Dense)               (None, 1)                 8

# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0
# _________________________________________________________________

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=100, batch_size=1)

# #4. 평가, 예측
# loss = model.evaluate(x,y)
# results = model.predict([[10, 1.3, 0]])
# print('로스 : ', loss)
# print('[10,1,3,0]의 예측값 : ', results)