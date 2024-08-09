# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

# y는 T(degC)로 잡아라. 144개

# 자르는 거는 맘대로

# 31.12.2016 00:10:00 부터
# 01.01.2017 00:00:00 까지

# 맞춰라!!! 1, 2, 3등 상 줌. 일요일 12시 59분까지

# LMSE로

# y의 shape는 (n, 144)
# 프레딕은 (1, 144)

# 소스코드와 가중치를 제출 할 것.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import time
import numpy as np
import pandas as pd
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"    # 현아님이 알려줌. 이렇게 하면 터지는게 덜하다 함...

#1. 데이터
path = 'C:/ai5/_data/kaggle/jena/'

csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)

sample_submission_jena_csv = pd.read_csv(path + 'jena_sample_submission.csv', index_col=0)

# print(csv.shape)  # (420551, 14)

# print(csv.columns)
# Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')

csv = csv[:-144]

# print(csv.shape)    # (420407, 14) <- 144개를 없앰.

x1 = csv.drop(['T (degC)'], axis=1)  # (420407, 13) <- T (degC) 없앰.

y1 = csv['T (degC)']

y3 = csv.tail(144)
y3 = y3['T (degC)']

# print(x1.shape) # (420407, 13)
# print(y1.shape) # (420407,)

size = 144

def split_x(dataset, size): 
    aaa = []
    for i in range(len(dataset) - size + 1):   
        subset = dataset[i : (i + size)]
        aaa.append(subset)                 
    return np.array(aaa)

x2 = split_x(x1, size)  

y2 = split_x(y1, size)

# print(x2.shape)   # 
# print(y2.shape)   # 

x = x2[:-1, :] 
y = y2[1:]

# print(x.shape)  # (420263, 144, 13) -> x는 맨 뒤에를 날리고.
# print(y.shape)  # (420263, 144)     -> y는 맨 앞을 날렸음.

# print(x2[-2])
# print(x[-1]) <- 잘 잘렸는지 확인해 봄.

# exit()

x_test2 = x[-1] # (144, 13, 1)

x_test2 = np.array(x_test2).reshape(1, 144, 13) 

# print(x_test2.shape)    # (1, 144, 13)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    shuffle= True,
                                                    random_state=123)


# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) 
# (378236, 144, 13) (42027, 144, 13) (378236, 144) (42027, 144)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

# print(x_train.shape, x_test.shape)  # (378236, 1872) (42027, 1872)

# exit()
scaler = StandardScaler() # MinMaxSc aler, StandardScaler, MaxAbsScaler, RobustScaler

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = np.reshape(x_train, (x_train.shape[0], 144, 13))
x_test = np.reshape(x_test, (x_test.shape[0], 144, 13))

# print(x_train.shape, x_test.shape)  # (378236, 144, 13) (42027, 144, 13)

# exit()

# 2. 모델
model = Sequential()
model.add(LSTM(16, return_sequences=True, input_shape=(144, 13))) # timesteps, features / 'tanh' / activation='tanh'
model.add(LSTM(32)) # activation='relu'
# model.add(Dropout(0.01)) # 드롭아웃해도 괜츈.
model.add(Dense(64)) 
model.add(Dense(128))
model.add(Dense(64)) 
model.add(Dense(32)) 
model.add(Dense(16)) 
model.add(Dense(144))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
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

model.fit(x_train, y_train, epochs=1000, batch_size=1024, validation_split=0.2, verbose=1, callbacks=[es, mcp])  # validation_split=0.1, mcp

end_time = time.time()

#4. 평가, 예측
# print("==================== 2. MCP 출력 =========================")
# model = load_model('C:/ai5/_data/kaggle/jena/')
results = model.evaluate(x, y)  # results 결과 / evaluate 평가 / batch_size=300

y_pred = model.predict(x_test2) # batch_size=300

y_pred = np.array(y_pred).reshape(144, 1)  

##################################################################################
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse = RMSE(y3, y_pred)

# print("RMSE : ", rmse)
#################################################################################

sample_submission_jena_csv['T (degC)'] = y_pred

sample_submission_jena_csv.to_csv(path + "sample_submission_jena_0809_1948.csv")

print("range(144)개의 결과 : ", y_pred)

print("로스는 : " , results[0])
print("RMSE : ", rmse)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 결과 / 로스 / RMSE / 걸린시간
# 로스는 :  13.975616455078125 / RMSE :  2.4520129437924267                         <- 'tanh' 에포 10
# 로스는 :  15.752904891967773 / RMSE :  5.461226169423893  / 걸린시간 :  136.35 초  <- 'tanh' 에포 100
# 로스는 :  85.57618713378906  / RMSE :  6.667439609313422                          <- 'relu' 에포 100
# 로스는 :  19.58470344543457  / RMSE :  3.5267952634604147 / 걸린시간 :  126.29 초  <- LSTM / return_sequences=True, batch_size 400
# 로스는 :  71.01980590820312  / RMSE :  11.662906715346399 / 걸린시간 :  665.68 초  <- 에포 500 <- 에포 100 / LSTM 2번, 모델 30에 렐루 / 배치 300, 
# 로스는 :  140.01348876953125 / RMSE :  3.819226486621923  / 걸린시간 :  251.9 초   <- 에포 500/ 드롭아웃 0.01, 모델 레이어 5개/ 
## 로스는 :  3.1943178176879883 / RMSE :  2.099804326743509  / 걸린시간 :  359.7 초   <- 드롭아웃, tanh, relu
# 로스는 :  458.3397216796875  / RMSE :  9.10431036257554   / 걸린시간 :  657.55 초  <- StandardScaler scaler
# 로스는 :  458.3397216796875  / RMSE :  9.10431036257554   / 걸린시간 :  657.55 초  <- StandardScaler scaler/ 모델 레이어 4개
# 로스는 :  291.8194274902344  / RMSE :  4.50452887859604   / 걸린시간 :  492.27 초  <- StandardScaler scaler/ 모델 레이어 7개
# 로스는 :  95.8620376586914   / RMSE :  11.441690032354902 / 걸린시간 :  386.66 초  <- 레이어 6개, 마름모 모형, 드롭아웃 추가
# 로스는 :  161.09266662597656 / RMSE :  7.809196008569161  / 걸린시간 :  1632.56 초 <- 모델 레이어 4개 , 
# 로스는 :  267.6151123046875  / RMSE :  4.9349421384849075 / 걸린시간 :  1800.92 초 <- 혜지님 모델
