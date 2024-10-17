# keras29_ModelCheckPonit3.py 복사

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
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

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=13))    
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 10,
    verbose= 1,
    restore_best_weights= True )

########################### mcp 세이프 파일명 만들기 시작 ################
import datetime 
date = datetime.datetime.now()
print(date) # 2024-07-26 16:51:36.578483
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date) # 0726 / 0726_1654
print(type(date))

path = './_save/keras30_mcp/01_boston/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k30_boston', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    save_weights_only=False,
    filepath = filepath,
)

class LastCheckpointSaver(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.params['epochs'] - 1:
            self.model.save(filepath, overwrite=True)
            print(f"최종 모델 저장: {filepath}")

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, 
                 validation_split=0.2, verbose=1, callbacks=[es, mcp])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("로스 : ", loss)
    
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2) 
# print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 로스 :  13.901333808898926
# r2스코어 :  0.8719385433738247


# 세이브한 가중치
# 로스 :  9.888299942016602
# r2스코어 :  0.9089073022687204