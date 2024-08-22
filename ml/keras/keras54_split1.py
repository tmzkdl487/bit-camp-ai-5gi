from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU , LSTM 
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

a = np.array(range(1, 11))

size = 5
# print(a)        # [ 1  2  3  4  5  6  7  8  9 10]
# print(a.shape)  # (10,)

def split_x(dataset, size): 
    aaa = []
    for i in range(len(dataset) - size + 1):    # len은 문자열의 길이 반환하는 함수. / len( "abc" )  # 3 반환
        subset = dataset[i : (i + size)]
        aaa.append(subset)                      # append 추가하다
    return np.array(aaa)
    
bbb = split_x(a, size)
print(bbb)      

'''
# [[ 1  2  3  4  5] -> 5개의 벡터 형태로 나눠줌. 
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
# print(bbb.shape)    # (6, 5) 

x = bbb[:, :-1] 
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]] 
y = bbb[:, -1]  # [ 5  6  7  8  9 10] 
# print(x, y) 
# print(x.shape, y.shape) # (6, 4) (6,)

x = x.reshape(x.shape[0], x.shape[1], 1)

# 2. 모델
model = Sequential()
model.add(SimpleRNN(units=21, input_shape=(4, 1), activation='relu')) # timesteps, features
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

x_predict = np.array([7, 8, 9, 10]).reshape(1, 4, 1)   
y_pred = model.predict(x_predict)

print("[11]의 결과 : ", y_pred)

# loss:  [3.6925694075762294e-06, 0.0] / [11]의 결과 :  [[11.038919]]
'''