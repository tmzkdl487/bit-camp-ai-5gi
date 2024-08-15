# 성우 하이텍 19일 월요일 종가를 맞춰봐!!!

# 제한시간 18일 일요일 23:59까지

# 앙상블 반드시 할 것!!!

# RNN 계열, 또는 Conv1D 쓸것!!!!

# 외부 데이터 사용 가능

# 외부 데이터 사용시 c:\ai5\_data\중간고사데이터\

from tensorflow.keras.models import Sequential, Model  
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten,  Input, Concatenate , concatenate
from tensorflow.keras.layers import Bidirectional, Conv1D, MaxPool1D  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder

import time
import numpy as np
import pandas as pd

#1. 데이터
path = 'C:/ai5/_data/중간고사데이터/'

NAVER = pd.read_csv(path + "NAVER 240816.csv", index_col=0, thousands=",")    #  encoding="cp949" <- 엑셀파일 한글 깨질때 쓰면 좋음.
HYBE = pd.read_csv(path + "하이브 240816.csv", index_col=0,thousands=",")    # thousands="," <- ""하면 문자열이라서 인식이 안되서 써야됨.
SUNGWOO = pd.read_csv(path + "성우하이텍 240816.csv", index_col=0, thousands=",")

# print(NAVER.columns)

# exit()

# Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
#        '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], dtype='object')

# print(NAVER.shape, HYBE.shape, SUNGWOO.shape)  # (5390, 17) (948, 17) (7058, 17)

NAVER = NAVER.sort_values(by=['일자'], ascending = True)
HYBE = HYBE.sort_values(by=['일자'], ascending = True)
SUNGWOO = SUNGWOO.sort_values(by=['일자'], ascending = True)

# print(NAVER)

# exit()

train_dt = pd.to_datetime(NAVER.index, format = '%Y/%m/%d')

NAVER['day'] = train_dt.day
NAVER['month'] = train_dt.month
NAVER['year'] = train_dt.year
NAVER['dos'] = train_dt.dayofweek

# print(NAVER.head()) # 위에만 나옴.
# exit()

train_dt = pd.to_datetime(HYBE.index, format = '%Y/%m/%d')

HYBE['day'] = train_dt.day
HYBE['month'] = train_dt.month
HYBE['year'] = train_dt.year
HYBE['dos'] = train_dt.dayofweek

train_dt = pd.to_datetime(SUNGWOO.index, format = '%Y/%m/%d')

SUNGWOO['day'] = train_dt.day
SUNGWOO['month'] = train_dt.month
SUNGWOO['year'] = train_dt.year
SUNGWOO['dos'] = train_dt.dayofweek


# print(NAVER.shape, HYBE.shape, SUNGWOO.shape)   # (5390, 20) (948, 20) (7058, 20)

# exit()

encoder = LabelEncoder()
NAVER['전일비'] = encoder.fit_transform(NAVER['전일비']).astype(float)
HYBE['전일비'] = encoder.fit_transform(HYBE['전일비']).astype(float)
SUNGWOO['전일비'] = encoder.fit_transform(SUNGWOO['전일비']).astype(float)

# print(NAVER['전일비'])

# exit()

# exit()

x1_1 = NAVER.drop(['Unnamed: 6', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1) 
x2_3 = HYBE.drop(['Unnamed: 6', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1) 
y4 = SUNGWOO.drop(['Unnamed: 6', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1) 

# print(x1_1.shape, x2_3.shape)   # (5390, 11) (948, 11)   

# exit()

x1_1 = x1_1[4442:]   # 네이버
x2_3 = x2_3          # 하이브

# print(x1_1.shape, x2_3.shape)   # (948, 11) (948, 11)

# exit()

y4 = y4[6110:]
# print(y4.shape) # (948, 11)

# exit()

y3 = y4['종가']  # 성우하이텍

# print(y3.shape) # (948,)

# exit()

x1_1test = x1_1.tail(20)

x2_3test = x2_3.tail(20)

# print(x1_1test.shape, x2_3test.shape)   # (20, 11) (20, 11)

# exit()

x1_1test = np.array(x1_1test).reshape(1, 20, 11)

x2_3test = np.array(x2_3test).reshape(1, 20, 11)

# print(x1_1test.shape, x2_3test.shape)   # (1, 20, 11) (1, 20, 11)

# exit()

size = 20

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):  
        subset = dataset[i : (i + size)]
        aaa.append(subset)                
    return np.array(aaa)

x1_2 = split_x(x1_1, size)  
x2_2 = split_x(x2_3, size) 
y2 = split_x(y3, size)

# print(x1_2.shape, x2_2.shape, y2.shape) # (919, 30, 21) (919, 30, 21) (919, 30)

# exit()

x1 = x1_2[:-1]
x2 = x2_2[:-1]
y = y2[1:]

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.75,
                                                    shuffle= True, random_state=11)

# print(x1_train.shape, x2_train.shape, y_train.shape)    # (881, 20, 11) (881, 20, 11) (881, 20)

x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2]))

x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2]))

# print(x1_train.shape, x1_test.shape) # (881, 220) (47, 220)
# print(x2_train.shape, x2_test.shape) # (881, 220) (47, 220)

# exit()
scaler = MaxAbsScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

scaler.fit(x1_train)
x_train = scaler.transform(x1_train)
x_test = scaler.transform(x1_test)

scaler.fit(x2_train)
x_train = scaler.transform(x2_train)
x_test = scaler.transform(x2_test)

x1_train = np.reshape(x1_train, (x1_train.shape[0], 20, 11))
x1_test = np.reshape(x1_test, (x1_test.shape[0], 20, 11))

x2_train = np.reshape(x1_train, (x2_train.shape[0], 20, 11))
x2_test = np.reshape(x1_test, (x2_test.shape[0], 20, 11))

# print(x1_train.shape, x1_test.shape) # (881, 20, 11) (47, 20, 11)
# print(x2_train.shape, x2_test.shape) # (881, 20, 11) (47, 20, 11)

# exit()

# 2-1. 모델
input1 = Input(shape=(20, 11))
dense1 = Bidirectional(LSTM(128, return_sequences=True, name='bit1'))(input1)
dense2 = Bidirectional(LSTM(64, return_sequences=True, activation = 'relu', name='bit2'))(dense1)
dense3 = Bidirectional(LSTM(64, activation = 'relu', name='bit3'))(dense2)
dense4 = Dense(32, activation = 'relu', name='bit4')(dense3) 
dense5 = Dense(32, activation = 'relu', name='bit5')(dense4)
dense6 = Dense(16, activation = 'relu', name='bit6')(dense5) 
output1 = Dense(20, activation = 'relu', name='bit7')(dense6)

# 2-2. 모델
input11 = Input(shape=(20, 11))   # size, x2.shape[-1]
dense11 = Bidirectional(LSTM(128, return_sequences=True, name='bit11'))(input11)
dense12 = Bidirectional(LSTM(64, return_sequences=True, activation = 'relu', name='bit31'))(dense11)
dense13 = Bidirectional(LSTM(64, activation = 'relu', name='bit32'))(dense12)
dense14 = Dense(32, activation = 'relu', name='bit33')(dense13)
dense15 = Dense(16, activation = 'relu', name='bit34')(dense14)
output11 = Dense(20, activation = 'relu', name='bit35')(dense15)

# 2-3. 합체!!!
merge1 = Concatenate(name='mg1')([output1, output11])
merge2 = Dense(64, activation='relu', name='mg2')(merge1) 
merge3 = Dense(32, activation='relu', name='mg3')(merge2)
merge4 = Dense(32, activation='relu', name='mg4')(merge3)
merge5 = Dense(32, activation='relu', name='mg5')(merge4)
last_output = Dense(20, name='last')(merge5)

model = Model(inputs=[input1, input11], outputs=last_output)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
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

path = 'C:/ai5/_save/중간고사가중치/'
filename = '{epoch:04d}-{loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k63_성우하이텍', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True,
    filepath = filepath,
)

model.fit([x1_train, x2_train], y_train, validation_split=0.2, epochs=1000, batch_size=32, verbose=1, callbacks=[es, mcp])  

end_time = time.time()

#4. 평가, 예측
# print("==================== 2. MCP 출력 =========================")
# model = load_model('C:/ai5/_save/중간고사가중치/')

loss = model.evaluate([x1_test, x2_test], y_test)

y_pred = model.predict([x1_1test, x2_3test])

print("성우하이텍 종가 : ", y_pred[0][0])
print("로스는 : " , loss[0])
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 성우하이텍 종가 :  9139.899  / 로스는 :  3427043.75  / 걸린시간 :  215.54 초
# 성우하이텍 종가 :  9233.927  / 로스는 :  3520759.25  / 걸린시간 :  131.6 초
# 성우하이텍 종가 :  8249.771  / 로스는 :  3337695.25  / 걸린시간 :  97.68 초  <- 칼럼 9개 없앰.
# 성우하이텍 종가 :  6671.0225 / 로스는 :  2978560.75  / 걸린시간 :  94.49 초  <- 100 에포
# 성우하이텍 종가 :  9613.507  / 로스는 :  3159812.75  / 걸린시간 :  97.29 초  <- 200 에포
# 성우하이텍 종가 :  8076.276  / 로스는 :  4182997.0   / 걸린시간 :  53.32 초  <- 150 에포
# 성우하이텍 종가 :  4286.9727 / 로스는 :  3132246.5   / 걸린시간 :  237.71 초 <- StandardScaler/ patience = 20/ 애포 250
# 성우하이텍 종가 :  4914.3203 / 로스는 :  3780194.75  / 걸린시간 :  64.19 초  <- StandardScaler/ patience = 20/ 애포 1000
# 성우하이텍 종가 :  7279.3433 / 로스는 :  3565499.5   / 걸린시간 :  87.05 초  <- MinMaxScaler/ 애포 1000
# 성우하이텍 종가 :  6716.52   / 로스는 :  3485220.0   / 걸린시간 :  65.06 초  <- MinMaxScaler/ 애포 1000 2
# 성우하이텍 종가 :  5817.1    / 로스는 :  3856193.25  / 걸린시간 :  84.17 초  <- MinMaxScaler/ patience = 20 / 애포 1000 3
# 성우하이텍 종가 :  7508.9375 / 로스는 :  3138083.75  / 걸린시간 :  68.7 초   <- MaxAbsScaler/ patience = 20 / 애포 1000 1
# 성우하이텍 종가 :  5193.311  / 로스는 :  3824727.25  / 걸린시간 :  42.74 초  <- MaxAbsScaler/ patience = 20 / 애포 1000 2
# 성우하이텍 종가 :  5303.411  / 로스는 :  3042239.25  / 걸린시간 :  80.62 초  <- MaxAbsScaler/ patience = 20 / 애포 1000 3
# 성우하이텍 종가 :  4574.9146 / 로스는 :  3418815.0   / 걸린시간 :  102.95 초 <- RobustScaler/ patience = 20 / 애포 1000 1
# 성우하이텍 종가 :  4642.65   / 로스는 :  3511490.0   / 걸린시간 :  83.64 초  <- RobustScaler/ patience = 20 / 애포 1000 2
# 성우하이텍 종가 :  6503.002  / 로스는 :  3659655.5   / 걸린시간 :  160.17 초 <- MaxAbsScaler/ patience = 30 / 에포 1000 
# 성우하이텍 종가 :  4756.041  / 로스는 :  2556325.75  / 걸린시간 :  153.52 초 <- train_size=0.9/ patience = 30 / epochs=1000
# 성우하이텍 종가 :  4711.7227 / 로스는 :  2628403.25  / 걸린시간 :  105.81 초 <- train_size=0.9/ patience = 30 / epochs=1000 2
# 성우하이텍 종가 :  5939.591  / 로스는 :  2635708.25  / 걸린시간 :  92.03 초  <- train_size=0.75, 
# 성우하이텍 종가 :  3945.5645 / 로스는 :  2742415.25  / 걸린시간 :  119.9 초  <- epochs=100000
# 성우하이텍 종가 :  4076.5662 / 로스는 :  3397433.0   / 걸린시간 :  30.12 초  <- epochs=100000 2트
# 성우하이텍 종가 :  12028.7705/ 로스는 :  2540818.0   / 걸린시간 :  151.8 초  <- 에포 1000 1트
# 성우하이텍 종가 :  5496.293  / 로스는 :  2863021.0   / 걸린시간 :  163.08 초 <- 2트
# 성우하이텍 종가 :  6007.0703 / 로스는 :  2584152.25  / 걸린시간 :  69.28 초  <- 100에포
# 성우하이텍 종가 :  6407.633  / 로스는 :  2576973.25  / 걸린시간 :  127.49 초 <- epochs=100, batch_size=3
# 성우하이텍 종가 :  4969.2563 / 로스는 :  2885925.0   / 걸린시간 :  30.3 초   <- batch_size=32  
# 성우하이텍 종가 :  6214.995  / 로스는 :  2454526.25  / 걸린시간 :  27.24 초  <- 2트    
# 성우하이텍 종가 :  6988.488  / 로스는 :  2285381.75  / 걸린시간 :  46.86 초  <- 3트    
# 성우하이텍 종가 :  7384.197  / 로스는 :  2362746.0   / 걸린시간 :  20.97 초  <- epochs=1000, batch_size=32   
# 성우하이텍 종가 :  5833.184  / 로스는 :  2356325.0   / 걸린시간 :  22.78 초  <- epochs=1500, batch_size=32    
# 성우하이텍 종가 :  6654.432  / 로스는 :  1901162.625 / 걸린시간 :  21.74 초  <- epochs=1000/ 모델 200으로 바꿈. 
# 성우하이텍 종가 :  17.971193 / 로스는 :  50410576.0  / 걸린시간 :  288.39 초 <- 모델에 0을 넣었더니...
# 성우하이텍 종가 :  6624.9424 / 로스는 :  2336430.0   / 걸린시간 :  14.7 초   <- 모델 100대 넣음
# 성우하이텍 종가 :  6357.6675 / 로스는 :  2362738.5   / 걸린시간 :  33.51 초
# 성우하이텍 종가 :  7747.948  / 로스는 :  1402759.125 / 걸린시간 :  332.52 초 <- 모델 1000대로 넣음.
# 성우하이텍 종가 :  6886.9824 / 로스는 :  1491429.375 / 걸린시간 :  179.02 초 <- 2트
# 성우하이텍 종가 :  5629.1333 / 로스는 :  1857386.75  / 걸린시간 :  94.61 초
# 성우하이텍 종가 :  5629.1333 / 로스는 :  1857386.75  / 걸린시간 :  94.61 초
# 성우하이텍 종가 :  4320.13   / 로스는 :  1061189.0   / 걸린시간 :  266.28 초 <- Bidirectional 넣고 LSTM 2개 넣음.
# 성우하이텍 종가 :  8931.319 / 로스는 :  446973.3125  / 걸린시간 :  501.63 초 <- LSTM 3개 넣어봄.
# 성우하이텍 종가 :  4218.858 / 로스는 :  1366883.125  / 걸린시간 :  317.83 초 <- LSTM 3개 넣어봄.2
# 성우하이텍 종가 :  3217.9   / 로스는 :  1856707.875  / 걸린시간 :  209.02 초 <- LSTM 3개 넣어봄.3
# 성우하이텍 종가 :  3029.8418 / 로스는 :  1977746.125 / 걸린시간 :  189.18 초 <- LSTM 3개 넣어봄.4
#    
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          