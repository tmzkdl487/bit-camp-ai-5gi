from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import sklearn as sk
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target 

print(x)    
print(y)
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=8989)

scaler = MaxAbsScaler() # MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(251, input_dim=10))
model.add(Dense(141))
model.add(Dense(171))
model.add(Dense(14))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True,
)

########################### mcp 세이프 파일명 만들기 시작 ################
import datetime 
date = datetime.datetime.now()
print(date) # 2024-07-26 16:51:36.578483
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date) # 0726 / 0726_1654
print(type(date))

path = './_save/keras30_mcp/03_diabetse/'
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

model.fit(x_train, y_train, epochs=1000, batch_size=32, 
          verbose=0, validation_split=0.2,
          callbacks= [es, mcp]
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print ("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# print("====================== hist =========================")
# print(hist)
# print("=================== hist.histroy ====================")
# print(hist.history)
# print("======================= loss =======================")
# print(hist.history['loss'])
# print("====================== val_loss ======================")
# print(hist.history['val_loss'])

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9, 6))
# plt.plot(hist.history['loss'], c = 'red', label = 'val_loss')
# plt.plot(hist.history['val_loss'], c = 'blue', label='val_loss')
# plt.legend(loc='upper right')
# plt.title('Diabetes')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# [실습] 맹그러봐 / R2 0.62 이상
# r2스코어 :  0.5189042303718636
# r2스코어 :  0.5665473691683233
# r2스코어 :  0.6007596386970162
# r2스코어 :  0.6006874672342544
# r2스코어 :  0.6151010366297509
# r2스코어 :  0.6180262183555738
# r2스코어 :  0.6196877963446061
# r2스코어 :  0.6211853813430838

# 그냥
# 로스 :  3531.4384765625 / r2스코어 :  0.4119794459086601

#[실습] MinMaxScaler 스켈링하고 돌려보기.
# 로스 :  3663.0 / r2스코어 :  0.39007311667484146

# [실습] StandardScaler 스켈링하고 돌려보기.
# 로스 :  3954.916748046875 / r2스코어 :  0.34146599025706104

# [실습] MaxAbsScaler 스켈링하고 돌려보기. 제일 좋음.
# 로스 :  3427.3046875 / r2스코어 :  0.4293187362085088

# [실습] RobustScaler 스켈링하고 돌려보기.
# 로스 :  4364.806640625 / r2스코어 :  0.2732151973621969