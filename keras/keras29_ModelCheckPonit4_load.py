# keras29_ModelCheckPonit3.py 복사

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import time

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=6666)

scaler = RobustScaler() # MinMaxScaler # StandardScale, MaxAbsScaler

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=13))    
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# es = EarlyStopping(
#     monitor= 'val_loss',
#     mode = 'min',
#     patience= 10,
#     verbose= 1,
#     restore_best_weights= True )

# mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_olny=True, 
#     filepath = './_save/keras29_mcp/keras29_mcp1.hdf5'
# )

# start_time = time.time()

# model.fit(x_train, y_train, epochs=1000, batch_size=16,   # hist는 히스토리를 줄인말이다.
#           verbose=1, callbacks = [es, mcp],
#           )
# end_time = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.h5')

print("=============================== 1. save_model 출력 ========================================")
model = load_model('./_save/keras29_mcp/keras29_3_save_model.h5')
loss = model.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss)
    
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2) 

print("=============================== 2. MCP 출력 ========================================")
model2 = load_model('./_save/keras29_mcp/keras29_mcp1.hdf5')
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss2)
    
y_predict2 = model.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print("r2스코어 : ", r2) 


################ 원래값 ######################
# 로스 :  11.266008377075195
# r2스코어 :  0.8838100282159193

# =============================== 1. save_model 출력 ========================================
# 로스 :  12.612658500671387
# r2스코어 :  0.8838100282159193

# =============================== 2. MCP 출력 ========================================
# 로스 :  11.266008377075195
# r2스코어 :  0.8838100282159193