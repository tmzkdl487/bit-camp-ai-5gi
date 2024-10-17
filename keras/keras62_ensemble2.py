# keras62_ensemble1.py 복사

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model      # 함수형 모델을 쓸 때 임포트 Model, Input을 해야한다.
from tensorflow.keras.layers import Dense, Input, Concatenate , concatenate # 대문자도 되고 소문자도 됨.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T
                      # 삼성종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511),
                        range(150, 250)]).transpose()
                      # 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301, 401),
                        range(77, 177), range(33, 133)]).T                      
# 거시기1, 거시기2, 거시기3, 거시기4
                      
y = np.array(range(3001, 3101)) # 한강의 화씨 온도.

x1_1 = np.array([range(100, 105), range(401, 406)]).T

x2_2 = np.array([range(201, 206), range(511, 516),range(250, 255)]).transpose()

x3_3 = np.array([range(100, 105), range(401, 406), range(177, 182), range(133, 138)]).T

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, x3_datasets, y, train_size=0.95,
                                                    shuffle= True, random_state=3)


# print(x3_train.shape, x3_test.shape)

# (95, 4) (5, 4)

# exit()

# 2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation = 'relu', name='bit1')(input1)
dense2 = Dense(20, activation = 'relu', name='bit2')(dense1)
dense3 = Dense(30, activation = 'relu', name='bit3')(dense2)
dense4 = Dense(20, activation = 'relu', name='bit4')(dense3)
output1 = Dense(10, activation = 'relu', name='bit5')(dense4)

# 2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(100, activation = 'relu', name='bit11')(input11)
dense21 = Dense(200, activation = 'relu', name='bit21')(dense11)
output11 = Dense(100, activation = 'relu', name='bit31')(dense21)

# 2-3. 모델
input12 = Input(shape=(4,))
dense12 = Dense(100, activation = 'relu', name='bit12')(input12)
dense22 = Dense(200, activation = 'relu', name='bit22')(dense12)
output12 = Dense(100, activation = 'relu', name='bit32')(dense22)

# 2-4. 합체!!!
merge1 = Concatenate(name='mg1')([output1, output11, output12])
merge2 = Dense(10, activation='relu', name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
merge4 = Dense(10, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)

model = Model(inputs=[input1, input11, input12], outputs=last_output)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights= True
)

########################### mcp 세이프 파일명 만들기 시작 ################
import datetime
date = datetime.datetime.now()
# print(date) # 2024-07-26 16:51:36.578483
# print(type(date))
date = date.strftime("%m%d_%H%M")
# print(date) # 0726 / 0726_1654
# print(type(date))

path = 'C:/ai5/_save/keras62/'
filename = '{epoch:04d}-{loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k62_ensemble2', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True,
    filepath = filepath,
)

model.fit([x1_train, x2_train, x3_train], y_train, epochs=100, batch_size=3, verbose=1, callbacks=[es, mcp])  # validation_split=0.2

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)

y_pred = model.predict([x1_1, x2_2, x3_3])

print("[3001~3101]결과 : ", y_pred)
print("로스는 : " , loss[0])
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# [3001~3101]결과 :  [[3108.4883] /  [3114.4663] /  [3120.5999] /  [3126.963 ] /  [3133.5356]] / 
# 로스는 :  0.8730481266975403 / 걸린시간 :  5.99 초

# [3001~3101]결과 :  [[3104.3064] /  [3108.5212] /  [3112.7356] /  [3116.95  ] /  [3121.1643]] / 
# 로스는 :  0.00033230782719329 / 걸린시간 :  18.09 초