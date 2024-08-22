from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import sklearn as sk
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=34)

scaler = StandardScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

'''
#2. 모델 구성
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=8))
model.add(Dense(30, activation='relu',))
model.add(Dense(30, activation='relu',))
model.add(Dense(20, activation='relu',))
model.add(Dense(3, activation='relu',))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True)

########################### mcp 세이프 파일명 만들기 시작 ################
import datetime 
date = datetime.datetime.now()
print(date) # 2024-07-26 16:51:36.578483
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date) # 0726 / 0726_1654
print(type(date))

path = './_save/keras30_mcp/02_california/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k29_', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=64,
                 verbose=1, validation_split=0.2,
                 callbacks= [es, mcp]
                 )
end_time = time.time()
'''

#4. 평가, 예측
print("==================== 2. MCP 출력 =========================")
model = load_model('./_save/keras30_mcp/02_california/k29_0729_0915_0056-1.326028.hdf5')
loss = model.evaluate(x_test, y_test)
print ("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)
# print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 세이브한 가중치
# # 로스 :  0.2662697434425354
# r2스코어 :  0.8021907608037689

# 로스 :  0.2662697434425354
# r2스코어 :  0.8021907608037689



