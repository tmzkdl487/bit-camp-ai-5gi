# 모델 구성해서 가중치까지 세이브할 것

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터

start_time1 = time.time()

np_path = 'c:/ai5/_data/_save_npy/'

x_train = np.load(np_path + 'keras45_gender_04_x_train.npy')
y_train = np.load(np_path + 'keras45_gender_04_y_train.npy')
x_test = np.load(np_path + 'keras45_gender_04_x_test.npy')
y_test = np.load(np_path + 'keras45_gender_04_y_test.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, 
                                                    shuffle= True,
                                                    random_state=911)


end_time1 = time.time()

#2. 모델
model = Sequential()
model.add(Conv2D(32, 2, activation='relu', input_shape = (100, 100, 3), padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(128, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])   # acc넣어야 분류일 경우 잘 맞는지 확인할 수 있음.
start_time2 = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
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
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k45_gender', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( 
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.25, verbose=1, callbacks=[es, mcp])

end_time2 = time.time()

#4. 평가, 예측
# print("==================== 2. MCP 출력 =========================")
# model = load_model('C:/ai5/_data/image/brain/k41_brain0805_1223_0081-0.014112.hdf5')
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16)  

y_pred = model.predict(x_test, batch_size=16)

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print(" 데이터 걸린시간 : ", round(end_time1 - start_time1, 2), "초")
print(" 걸린시간 : ", round(end_time2 - start_time2, 2), "초")

# 세이브한 가중치 값
# 로스는 : 0.36696934700012207 / ACC :  0.846 /  데이터 걸린시간 :  1.38 초 /  걸린시간 :  358.26 초
# 로스는 :  0.34271466732025146 / ACC :  0.858 /  데이터 걸린시간 :  1.43 초 /  걸린시간 :  328.62 초
# 로스는 :  0.29835057258605957 / ACC :  0.872 /  데이터 걸린시간 :  1.39 초 /  걸린시간 :  471.65 초