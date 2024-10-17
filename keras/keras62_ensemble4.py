# keras62_ensemble3.py 복사

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model      # 함수형 모델을 쓸 때 임포트 Model, Input을 해야한다.
from tensorflow.keras.layers import Dense, Input, Concatenate , concatenate # 대문자도 되고 소문자도 됨.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T
                      # 삼성종가, 하이닉스 종가
                      
y1 = np.array(range(3001, 3101)) # 한강의 화씨 온도.

y2 = np.array(range(13001, 13101)) # 비트코인 가격

x1_1 = np.array([range(100, 105), range(401, 406)]).T

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        x1_datasets, y1, y2, train_size=0.95, shuffle= True, random_state=3)


# print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_trian.shape)

# (95, 4) (5, 4)

# exit()


# 2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(32, activation = 'relu', name='bit1')(input1)
dense2 = Dense(64, activation = 'relu', name='bit2')(dense1)
dense3 = Dense(128, activation = 'relu', name='bit3')(dense2)
dense4 = Dense(64, activation = 'relu', name='bit4')(dense3)
dense5 = Dense(32, activation = 'relu', name='bit5')(dense4)
output1 = Dense(1, activation = 'relu', name='bit6')(dense5)

last_output = Dense(1, name='last')(output1)
last_output2 = Dense(1, name='last2')(output1) 

model = Model(inputs= input1, outputs = [last_output, last_output2])

# model.summary()

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
filepath = "".join([path, 'k62_ensemble4', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True,
    filepath = filepath,
)

model.fit(x1_train, [y1_train, y2_train], epochs=2000, batch_size=2, verbose=1, callbacks=[mcp])  #  es, validation_split=0.2

end_time = time.time()



#4. 평가, 예측
# print("==================== 2. MCP 출력 =========================")
# model = load_model('C:/ai5/_save/keras62/k62/k62_ensemble40814_1715_1000-179403120.000000.hdf5')

loss = model.evaluate(x1_test, [y1_test, y2_test])

y_pred = model.predict(x1_1)

print("[3001~3101]결과 : ", y_pred)
print("로스는 : " , loss)
# print("걸린시간 : ", round(end_time - start_time, 2), "초")   

# [3001~3101]결과 :  [array([[4267.675 ], [4278.8394], [4290.003 ], [4301.167 ], [4312.332 ]], dtype=float32), array([[0.],
# [0.], [0.], [0.],  [0.]], dtype=float32)]
# 로스는 :  [171495600.0, 768982.3125, 170726624.0, 0.0, 0.0]
# 걸린시간 :  1.49 초 

# [3001~3101]결과 :  [array([[31.989693], [31.989693],  [31.989693], [31.989693],  [31.989693]], type=float32), array([[31.990456],
# [31.990456],   [31.990456],  [31.990456],  [31.990456]], dtype=float32)]
# 로스는 :  [179099152.0, 9207478.0, 169891680.0, 0.0, 0.0]   

#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

# last_output = Dense(1, name='last')(merge3)
# last_output2 = Dense(1, name='last2')(merge3) <- 사영님 방법

# 2-5. 분기1
dense51 = Dense(10, activation = 'relu', name='bit51')(middle_output)
dense52 = Dense(20, activation = 'relu', name='bit52')(dense51)
dense53 = Dense(20, activation = 'relu', name='bit53')(dense52)
output_1 = Dense(1, activation = 'relu', name='bit54')(dense53)

# 2-6. 분기2
dense61 = Dense(16, activation = 'relu', name='bit61')(middle_output)
dense62 = Dense(26, activation = 'relu', name='bit62')(dense61)
output_2 = Dense(1, activation = 'relu', name='outpoout2')(dense62)

model = Model(inputs=[input1, input11, input12], outputs = [output_1, output_2])

# model.summary()

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

path = 'C:/ai5/_data/kaggle/jena/'
filename = '{epoch:04d}-{loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k55_jena', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True,
    filepath = filepath,
)

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=100, batch_size=16, verbose=1, callbacks=[es, mcp])  # validation_split=0.2

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])

y_pred = model.predict([x1_1, x2_2, x3_3])

print("[3001~3101]결과 : ", y_pred[0])
print("[3001~3101]결과 : ", y_pred[1])
print("로스는 : " , loss)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# [3001~3101]결과 :  [array([[3105.6274], [3110.6328], [3115.753 ], [3121.0752], [3126.3896]], dtype=float32), array([[13145.529],
# [13165.739], [13186.5], [13208.252], [13230.005]], dtype=float32)]
# 로스는 :  12.31168270111084

# [3001~3101]결과 :  [[3167.476 ] /  [3177.5564] /  [3187.6382] /  [3197.7178] /  [3207.7976]] / 
# [3001~3101]결과 :  [[13822.011] /  [13866.999] /  [13911.988] /  [13956.978] /  [14001.965]]
# 로스는 :  2759909.5 / 걸린시간 :  1.64 초

# 로스는 :  [9402639.0, 9402629.0, 9.641108512878418, 0.0, 0.0] 
# <- 로스가 많이 나오는 이유는 첫번째 로스는 두번째 세번째 로스를 합친 것이다.
# 첫번째는 전체 로스, 두번째는 y1에 대한 로스, 세번째는 y2에 대한 로스
# 9402639.0 = 9402629.0 + 9.641108512878418 / 여기에 
# 로스는 :  [9402639.0, 9402629.0, 9.641108512878418, 0.0, 0.0]
# 메트릭스 넣으면 0.0, 0.0 까지 로스가 5개 나옴.