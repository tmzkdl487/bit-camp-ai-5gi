import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model      # 함수형 모델을 쓸 때 임포트 Model, Input을 해야한다.
from keras.layers import Dense, Input, Concatenate , concatenate # 대문자도 되고 소문자도 됨.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T
                      # 삼성종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511),
                        range(150, 250)]).transpose()
                      # 원유, 환율, 금시세
                      
y = np.array(range(3001, 3101)) # 한강의 화씨 온도.

# print(x1_datasets.shape, x2_datasets.shape) # (2, 100) (3,) -> .T써서 (100, 2) (3,)

# exit()

x1_1 = np.array([range(100, 105), range(401, 406)]).T

x2_2 = np.array([range(201, 206), range(511, 516),range(250, 255)]).transpose()

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, train_size=0.95,
                                                    shuffle= True, random_state=3)

# print(x1_train.shape, x2_train.shape, y_train.shape)
# 

# 2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(16, activation = 'relu', name='bit1')(input1)
dense2 = Dense(32, activation = 'relu', name='bit2')(dense1)
dense3 = Dense(64, activation = 'relu', name='bit3')(dense2)
dense4 = Dense(32, activation = 'relu', name='bit4')(dense3)
output1 = Dense(16, activation = 'relu', name='bit5')(dense4)
# model1 = Model(inputs = input1, outputs = output1)

# model1.summary()

#  compute capability: 8.6
# _________________________________________________________________
# =================================================================
#  input_1 (InputLayer)        [(None, 2)]               0

#  bit1 (Dense)                (None, 10)                30

#  bit2 (Dense)                (None, 20)                220

#  bit3 (Dense)                (None, 30)                630

#  bit4 (Dense)                (None, 40)                1240

#  bit5 (Dense)                (None, 50)                2050

# =================================================================
# Total params: 4,170
# Trainable params: 4,170
# Non-trainable params: 0
# _________________________________________________________________

# exit()

# 2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(50, activation = 'relu', name='bit11')(input11)
dense21 = Dense(100, activation = 'relu', name='bit21')(dense11)
output11 = Dense(50, activation = 'relu', name='bit31')(dense21)
# model2 = Model(inputs = input11, outputs = output11)

# model1.summay()

# 2-3. 합체!!!
# merge1 = concatenate([output1, output11], name='mg1')   # merge  병합하다.
# merge2 = Dense(7, activation='relu', name='mg2')(merge1)
# merge3 = Dense(20, name='mg3')(merge2)
# last_output = Dense(1, name='last')(merge3)

# model = Model(inputs=[input1, input11], outputs=last_output)

# model.summary()

# Model: "model" / merge1 = concatenate
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 2)]          0           []

#  bit1 (Dense)                   (None, 10)           30          ['input_1[0][0]']

#  bit2 (Dense)                   (None, 20)           220         ['bit1[0][0]']

#  input_2 (InputLayer)           [(None, 3)]          0           []

#  bit3 (Dense)                   (None, 30)           630         ['bit2[0][0]']

#  bit11 (Dense)                  (None, 100)          400         ['input_2[0][0]']

#  bit4 (Dense)                   (None, 40)           1240        ['bit3[0][0]']

#  bit21 (Dense)                  (None, 200)          20200       ['bit11[0][0]']

#  bit5 (Dense)                   (None, 50)           2050        ['bit4[0][0]']

#  bit31 (Dense)                  (None, 300)          60300       ['bit21[0][0]']

#  mg1 (Concatenate)              (None, 350)          0           ['bit5[0][0]',
#                                                                   'bit31[0][0]']

#  mg2 (Dense)                    (None, 7)            2457        ['mg1[0][0]']

#  mg3 (Dense)                    (None, 20)           160         ['mg2[0][0]']

#  last (Dense)                   (None, 1)            21          ['mg3[0][0]']

# ==================================================================================================
# Total params: 87,708
# Trainable params: 87,708
# Non-trainable params: 0
# __________________________________________________________________________________________________
# 2-3. 합체!!!
merge1 = Concatenate(name='mg1')([output1, output11])
merge2 = Dense(10, activation='relu', name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
merge4 = Dense(10, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)

model = Model(inputs=[input1, input11], outputs=last_output)

# model.summary()
# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 2)]          0           []

#  bit1 (Dense)                   (None, 10)           30          ['input_1[0][0]']

#  bit2 (Dense)                   (None, 20)           220         ['bit1[0][0]']

#  input_2 (InputLayer)           [(None, 3)]          0           []

#  bit3 (Dense)                   (None, 30)           630         ['bit2[0][0]']

#  bit11 (Dense)                  (None, 100)          400         ['input_2[0][0]']

#  bit4 (Dense)                   (None, 40)           1240        ['bit3[0][0]']

#  bit21 (Dense)                  (None, 200)          20200       ['bit11[0][0]']

#  bit5 (Dense)                   (None, 50)           2050        ['bit4[0][0]']

#  bit31 (Dense)                  (None, 300)          60300       ['bit21[0][0]']

#  mg1 (Concatenate)              (None, 350)          0           ['bit5[0][0]',
#                                                                   'bit31[0][0]']

#  mg2 (Dense)                    (None, 7)            2457        ['mg1[0][0]']

#  mg3 (Dense)                    (None, 20)           160         ['mg2[0][0]']

#  last (Dense)                   (None, 1)            21          ['mg3[0][0]']

# ==================================================================================================
# Total params: 87,708
# Trainable params: 87,708
# Non-trainable params: 0
# __________________________________________________________________________________________________

# 맹그러봐!!!

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
filepath = "".join([path, 'k62_ensemble1', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True,
    filepath = filepath,
)

model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=3, verbose=1, callbacks=[es, mcp])  # validation_split=0.2

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)

y_pred = model.predict([x1_1, x2_2])

print("[3001~3101]결과 : ", y_pred)
print("로스는 : " , loss[0])
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# [3001~3101]결과 :  [[3522.7166] /  [3281.7598] /  [2716.4192] /  [3253.9573] /  [3550.518 ]] /
# 로스는 :  111062.359375 / 걸린시간 :  2.42 초 <- 100 에포

# [3001~3101]결과 :  [[3563.1023] /  [3316.4844] /  [2737.9023] /  [3288.0298] /  [3591.5637]] /
# 로스는 :  129710.1015625 / 걸린시간 :  2.97 초 <- 1000에포

# [3001~3101]결과 :  [[3093.9976] /  [3067.9988] /  [3007.001 ] /  [3064.9988] /  [3096.9976]] /
# 로스는 :  3.1709671475255163e-06 / 걸린시간 :  29.69 초

# [3001~3101]결과 :  [[3093.947 ] /  [3067.968 ] /  [3006.94  ] /  [3064.9702] /  [3096.944 ]] / 
# 로스는 :  0.0022899031173437834  / 걸린시간 :  61.12 초

# 범위 지정함
# [3001~3101]결과 :  [[3102.0305] /  [3104.5369] /  [3107.7612]  /  [3111.512 ]  /  [3116.2026]] / 
# 로스는 :  0.0834910124540329 / 걸린시간 :  37.91 초

# [3001~3101]결과 :  [[3101.468 ] /  [3105.265 ] /  [3109.1653] /  [3113.0654] /  [3117.052 ]] /
# 로스는 :  5.664166450500488 / 걸린시간 :  123.56 초

# [3001~3101]결과 :  [[3100.708 ] /  [3101.7043] /  [3102.7002] /  [3103.6963] /  [3104.653 ]] / 
# 로스는 :  0.040746498852968216 / 걸린시간 :  203.1 초