# keras52_LSTM2_scale.py 복사

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, GRU , LSTM 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50,],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])                    

print(x.shape, y.shape) # (13, 3) (13,)

# exit()

# 아워너 80

# 시작!

# LSTM을 2개 이상 넣어봐라!!!! 
# -> return_sequences=True 를 넣으면 3차원 데이터를 받아서 2차원 데이터로 뱉어내는 RNN도 3차원 데이터를 넣을 수 있음.
# -> 그러나, RNN을 2번 넣었다고 해서 성능 장담하지 못함.
model = Sequential()
model.add(LSTM(21, return_sequences=True, input_shape=(3,1), activation='relu')) # timesteps, features
model.add(LSTM(20))
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

########################### mcp 세이프 파일명 만들기 시작 ################
import datetime 
date = datetime.datetime.now()
print(date) # 2024-07-26 16:51:36.578483
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date) # 0726 / 0726_1654
print(type(date))

path = './_save/keras35/'
filename = '{epoch:04d}-{loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k52_LSTM2', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( 
    monitor='loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x, y, epochs=1000, batch_size=8, verbose=1, callbacks=[es, mcp])  # validation_split=0.1,

#4. 평가, 예측
# print("==================== 2. MCP 출력 =========================")
# model = load_model('C:/ai5/_save/keras35/76.8/k52_LSTM20807_1737_0112-0.840497.hdf5')

results = model.evaluate(x, y)
print('loss: ', results)

x_predict = np.array([50,60,70]).reshape(1, 3, 1)   
y_pred = model.predict(x_predict)

print("[80]의 결과 : ", y_pred)

# loss:  [8.80083825904876e-05, 0.0] / [80]의 결과 :  [[79.958405]] <- 기현님 모델

# LSTM을 2개 넣은 것.
# loss:  [0.0004943051608279347, 0.0] / [80]의 결과 :  [[71.47763]]