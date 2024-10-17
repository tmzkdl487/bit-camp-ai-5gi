from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU , LSTM 
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]]).T #reshape(10,2) / T 

# x는 (N, 5, 2)
# y는 (N,) 형태로 맹그러
size = 6

def split_x(dataset, size): 
    aaa = []
    for i in range(len(dataset) - size + 1):    # len은 문자열의 길이 반환하는 함수. / len( "abc" )  # 3 반환
        subset = dataset[i : (i + size)]
        aaa.append(subset)                      # append 추가하다
    return np.array(aaa)
    
bbb = split_x(a, size)

x = bbb[:, :-1] 
y = bbb[:, -1, 0] 

print(bbb) 
# [[[ 1  9] 
#   [ 2  8]
#   [ 3  7]
#   [ 4  6]
#   [ 5  5]
#   [ 6  4]]

#  [[ 2  8]
#   [ 3  7]
#   [ 4  6]
#   [ 5  5]
#   [ 6  4]
#   [ 7  3]]

#  [[ 3  7]
#   [ 4  6]
#   [ 5  5]
#   [ 6  4]
#   [ 7  3]
#   [ 8  2]]

#  [[ 4  6]
#   [ 5  5]
#   [ 6  4]
#   [ 7  3]
#   [ 8  2]
#   [ 9  1]]

#  [[ 5  5]
#   [ 6  4]
#   [ 7  3]
#   [ 8  2]
#   [ 9  1]
#   [10  0]]]

# -> 우리가 원하는 데로 잘라짐. 
# 근데 6개의 벡터로 나눠짐. x는 (N, 5, 2)를 만들고 싶음. 그래서 마지막 열을 없애고 싶어서 [:, :-1] 을 넣어서 마지막 벡터를 없앰.
# y는 (N,) 형태로 만들고 싶어서 [:, -1, 0] 뒤에는 -1로 없애고, 앞에 0째 열만 가지고 옴.


# print('----------------')
# print(x)
# [[[1 9]
#   [2 8]
#   [3 7]
#   [4 6]
#   [5 5]]

#  [[2 8]
#   [3 7]
#   [4 6]
#   [5 5]
#   [6 4]]

#  [[3 7]
#   [4 6]
#   [5 5]
#   [6 4]
#   [7 3]]

#  [[4 6]
#   [5 5]
#   [6 4]
#   [7 3]
#   [8 2]]

#  [[5 5]
#   [6 4]
#   [7 3]
#   [8 2]
#   [9 1]]]
print(y)  # [ 6  7  8  9 10]

print(x.shape, y.shape) # (5, 5, 2) (5,)

# => 그래서 원하는 데로 x, y를 잘 만듬.

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
print('loss: ', results[0])

x_predict = np.array([[6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]).reshape(1, 5, 2)   
y_pred = model.predict(x_predict)

print("ACC : ", round(results[1], 3))
print("[11]의 결과 : ", y_pred)

# loss:  [0.2500000298023224, 0.0833333358168602] / [11]의 결과 :  [[4.623344]]
# loss:  9.166666984558105 / ACC :  0.083 / [11]의 결과 :  [[4.7077575]] <- .T했는데도 똑같아. 
# loss:  9.022187164031692e-11 / ACC :  0.0 / [11]의 결과 :  [[13.426356]]

