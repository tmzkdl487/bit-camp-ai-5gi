# 54_3 카피해서
# (N, 10, 1) -> (N, 5, 2)
# 맹그러봐

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU , LSTM 
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))  # 101부터 107을 찾아라

# 맹그러봐!!!

size = 11

# print(a.shape)  # (100,)

def split_x(dataset, size): 
    aaa = []
    for i in range(len(dataset) - size + 1):   
        subset = dataset[i : (i + size)]
        aaa.append(subset)                 
    return np.array(aaa)

bbb = split_x(a, size)   # (96, 4, 1) (96,)

# print(bbb)  

x = bbb[:, :-1] 
y = bbb[:, -1]

# print(x)
# print(y)
# print(x.shape, y.shape) # (90, 10) (90,)

x = x.reshape(x.shape[0], x.shape[1], 1)
x = x.reshape(90, 5, 2)
# print(x.shape, y.shape) # (90, 10, 1) (90,)


# 2. 모델
model = Sequential()
model.add(SimpleRNN(units=21, input_shape=(5, 2), activation='relu')) # timesteps, features
model.add(Dense(20)) 
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    patience = 30,
    restore_best_weights= True
)

model.fit(x, y, epochs=1000, batch_size=8, verbose=1, callbacks=[es])  # validation_split=0.1, mcp

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss: ', results)

x_predict = np.array(range(96, 106)).reshape(1, 5, 2)   
y_pred = model.predict(x_predict)

print("range(96, 106)의 결과 : ", y_pred)

# loss:  [3.609625309763942e-06, 0.0] / range(96, 106)의 결과 :  [[106.08006]]
# range(96, 106)의 결과 :  [[106.02538]]

