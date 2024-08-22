# keras51_RNN1.py 복사

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],]
             )

y = np.array([4,5,6,7,8,9,10,])

# print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)    # 3차원 데이터로 리쉐이프. / x = x.reshape(7, 3, 1) 
print(x.shape)  # 프린트 찍으니 (7, 3, 1) 나옴. 3-D tensor with shape (batch_size, timesteps, features)

#2. 모델
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3, 1))) 
# model.add(SimpleRNN(10, input_length=3, input_dim=1)) # 이렇게 써도 돌아감.

model.add(SimpleRNN(10, input_dim=1, input_length=3))   # 거꾸로 써도 돌아감. 가독성이 떨어져. 쓰고 싶으면 써!
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights= True
)

model.fit(x, y, epochs=1000, batch_size=8,
          validation_split=0.3, verbose=1, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss: ', results)

x_pred = np.array([8,9,10]).reshape(1, 3, 1)   # [[[8],[9],[10]]]
y_pred = model.predict(x_pred)
# (3, ) -> (1, 3, 1)

print("[8, 9, 10]의 결과 : ", y_pred)

# loss:  34.9900016784668 / [8, 9, 10]의 결과 :  [[1.4761065]]

# 11.00000만들기 SimpleRNN
# loss:  [0.40446189045906067, 0.0] / [8, 9, 10]의 결과 :  [[11.51098]]
# [8, 9, 10]의 결과 :  [[11.484153]]

# LSTM
# loss:  [0.31389737129211426, 0.0] / [8, 9, 10]의 결과 :  [[9.707025]]

# GRU
# loss:  [0.5979174375534058, 0.0] / [8, 9, 10]의 결과 :  [[10.350551]]