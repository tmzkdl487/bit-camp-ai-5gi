# keras56_Bidirectional2.py 복사

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from sklearn.model_selection import train_test_split
import time 

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50,],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])                    # 아워너 80

# Conv1D로 맹그러

# 시작!!!

# print(x.shape, y.shape, x_predict.shape)    # (13, 3) (13,) (3,)

x = x.reshape(x.shape[0], x.shape[1], 1)    
# print(x.shape)  # (13, 3, 1)

# scaler = StandardScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Conv1D(10, kernel_size=2, input_shape=(3, 1))) # timesteps, features
model.add(Conv1D(10, 2))
model.add(Flatten())
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

start_time = time.time()

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

end_time = time.time()

#4. 평가, 예측
# print("==================== 2. MCP 출력 =========================")
# model = load_model('C:/ai5/_save/keras35/76.8/k52_LSTM20807_1737_0112-0.840497.hdf5')

results = model.evaluate(x, y)
print('loss: ', results)

x_predict = np.array([50,60,70]).reshape(1, 3, 1)   
y_pred = model.predict(x_predict)

print("[80]의 결과 : ", y_pred)
print("데이터 걸린시간 : ", round(end_time - start_time, 2), "초")

# 아워너 80
# loss:  [510.03216552734375, 0.0] / [80]의 결과 :  [[14.160299]]
# loss:  [183.13307189941406, 0.0] / [80]의 결과 :  [[36.78913]] <- LSTM / 모델 좀 줄임. 
# loss:  [342.68316650390625, 0.0] / [80]의 결과 :  [[23.066053]] <- 모델 줄이니까 결과값 떨어짐.
# loss:  [467.4697570800781, 0.0] / [80]의 결과 :  [[16.4809]] <- SimpleRNN
# loss:  [280.90228271484375, 0.0] / [80]의 결과 :  [[29.808125]] <- GRU
# loss:  [163.75697326660156, 0.0] / 
# [80]의 결과 :  [[46.17838]]<- 드롭 좀 빼고 다시 돌림
# loss:  [320.77764892578125, 0.0] / [80]의 결과 :  [[24.304653]]
# loss:  [315.2862854003906, 0.0] / [80]의 결과 :  [[23.26021]]갯수 좀 줄임.
# loss:  [468.4090576171875, 0.0] / [80]의 결과 :  [[14.282044]]<- 모델 숫자 많이 줄임.
# loss:  [496.5750732421875, 0.0] / [80]의 결과 :  [[12.857337]]
# loss:  [467.27032470703125, 0.0] / [80]의 결과 :  [[14.658313]]
# loss:  [1.2006086111068726, 0.0] / [80]의 결과 :  [[70.6337]] <-  발리데이션 줄임.
# loss:  [4.465026378631592, 0.0] / [80]의 결과 :  [[68.671394]]
# loss:  [7.584999084472656, 0.0] / [80]의 결과 :  [[72.344]] <- 드롭아웃 1개 model.add(Dropout(0.1))
# loss:  [3.426185131072998, 0.0] /[80]의 결과 :  [[69.71652]] <- 그롭아웃 2개 model.add(Dropout(0.25))
# loss:  [4.3506574630737305, 0.0] / [80]의 결과 :  [[73.35883]]<- 맨처음 모델 128
# loss:  [4.491365432739258, 0.0] / [80]의 결과 :  [[74.142426]]
# loss:  [0.99642413854599, 0.0] / [80]의 결과 :  [[75.612236]]
# loss:  [0.7054460048675537, 0.0] / [80]의 결과 :  [[78.85535]]
# loss:  [0.2131989598274231, 0.0] / [80]의 결과 :  [[77.48037]]
# loss:  [8.258136749267578, 0.0] / [80]의 결과 :  [[85.74606]] <- 모델 적게하고 앞에 렐루 땔리니까 나옴.
# loss:  [8.80083825904876e-05, 0.0] / [80]의 결과 :  [[79.958405]]

####################### Bidirectional 넣어서 만들어보기.
# loss:  [0.00026091327890753746, 0.0] / [80]의 결과 :  [[79.96578]]

# Conv1 2개씀
# loss:  [1.044030341290636e-06, 0.0] / [80]의 결과 :  [[79.999435]] / 데이터 걸린시간 :  25.91 초

