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

# sample_submission_jena_csv = pd.read_csv(path + 'jena_sample_submission.csv', index_col=0)

# print(csv.shape)  # (420551, 14)

# print(csv.columns)
# Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')

train_dt = pd.DatetimeIndex(csv.index)

csv['day'] = train_dt.day
csv['month'] = train_dt.month
csv['year'] = train_dt.year
csv['hour'] = train_dt.hour
csv['dos'] = train_dt.dayofweek

# print(csv)

y3 = csv.tail(144)
y3 = y3['T (degC)']

# print(y3.shape) # (144,)

# exit()

csv = csv[:-144]

# print(csv.shape)    # (420407, 14) <- 144개를 없앰. / (420407, 19)

# print(y3.shape) # (144,)

# exit()
x1 = csv.drop(['T (degC)', 'max. wv (m/s)', 'max. wv (m/s)', 'wd (deg)',"year"], axis=1)  # (420407, 13) <- T (degC) 없앰, 'wd (deg)'

y1 = csv['T (degC)']

# print(x1.shape) # (420407, 13) / (420407, 17) / (420407, 15)
# print(y1.shape) # (420407,)    / (420407,)    / (420407,)

# exit()

size = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):  
        subset = dataset[i : (i + size)]
        aaa.append(subset)                
    return np.array(aaa)

x2 = split_x(x1, size)  

y2 = split_x(y1, size)

# print(x2.shape)   # (420264, 144, 15)
# print(y2.shape)   # (420264, 144)

# exit()

x = x2[:-1, :]
y = y2[1:]

# print(x.shape)  # (420263, 144, 13) -> x는 맨 뒤에를 날리고. / (420407, 17) / (420263, 144, 15)
# print(y.shape)  # (420263, 144)     -> y는 맨 앞을 날렸음.   / (420407,)    / (420263, 144)

# print(x2[-2])
# print(x[-1]) <- 잘 잘렸는지 확인해 봄.

# print(x.shape)  # (420263, 144, 15)
# print(y.shape)  # (420263, 144)

# exit()

x_test2 = x2[-1] # (144, 13, 1)

# print(x_test2.shape)    # (144, 17)

# exit()

x_test2 = np.array(x_test2).reshape(1, 144, 15)

# print(x_test2.shape)    # (1, 144, 13) / (1, 144, 17)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95,
                                                    shuffle= True,
                                                    random_state=3)


# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (378236, 144, 13) (42027, 144, 13) (378236, 144) (42027, 144) / (378236, 144, 17) (42027, 144, 17) (378236, 144) (42027, 144)

# exit()

# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

# print(x_train.shape, x_test.shape)  # (378236, 1872) (42027, 1872) / (378236, 2160) (42027, 2160)

# exit()
# scaler = StandardScaler() # MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = np.reshape(x_train, (x_train.shape[0], 144, 15))
# x_test = np.reshape(x_test, (x_test.shape[0], 144, 15))

# print(x_train.shape, x_test.shape)  # (378236, 144, 13) (42027, 144, 13) / (420407, 17) (420407,) / 

# exit()

# 2. 모델
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(144, 15)))   # timesteps, features / activation='tanh'
model.add(LSTM(120))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(370, activation='relu'))
model.add(Dropout(0.1)) 
model.add(Dense(320, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(288, activation='relu'))
model.add(Dense(144))


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

model.fit(x_train, y_train, epochs=200, batch_size=500, validation_split=0.2, verbose=1, callbacks=[es, mcp])  # validation_split=0.1, mcp

end_time = time.time()

#4. 평가, 예측
# print("==================== 2. MCP 출력 =========================")
# model = load_model('C:/ai5/_data/kaggle/jena/')
results = model.evaluate(x_test, y_test)  # results 결과 / evaluate 평가 / batch_size=300

y_pred = model.predict(x_test2) # batch_size=300

y_pred = np.array(y_pred).reshape(144, 1)  

##################################################################################
from sklearn.metrics import mean_squared_error

def RMSE(y3, y_pred):
    return np.sqrt(mean_squared_error(y3, y_pred))

rmse = RMSE(y3, y_pred)

# print("RMSE : ", rmse)
#################################################################################

# sample_submission_jena_csv['T (degC)'] = y_pred

# sample_submission_jena_csv.to_csv(path + "sample_submission_jena_0809_0804.csv")

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
# 로스는 :  217.58509826660156 / RMSE :  9.308505708023477  /  걸린시간 :  3709.53 초 
# ======================================================================================= 이전에거는 잘 못 돌림. 다시 시작...
# 로스는 :  0.07453043758869171 / RMSE :  12.694314125993746 / 걸린시간 :  2656.16 초 <- 바꾸고 돌림 / 에포 1000
# 로스는 :  0.12281683087348938 / RMSE :  4.917208498503314  / 걸린시간 :  694.59 초  <- 모델을 많이 넣고 돌리고 에포 100
# 로스는 :  0.2559919059276581  / RMSE :  8.906963987699898  / 걸린시간 :  331.66 초  <- scaler = RobustScaler()로 돌리기 1
# 로스는 :  0.1905936449766159  / RMSE :  15.01088269404153  / 걸린시간 :  428.77 초  <- scaler = RobustScaler()로 돌리기 2
# 로스는 :  2.8863322734832764  / RMSE :  3.5990894881671975 / 걸린시간 :  181.82 초  <- 기현님 모델
# 로스는 :  9.420337677001953   / RMSE :  3.5308723008702776 / 걸린시간 :  166.67 초  <- 1000에포
# 로스는 :  0.6138196587562561  / RMSE :  11.671496381443363 / 걸린시간 :  178.27 초  <- 기현님 모델에 scaler = RobustScaler()로 돌리기 / 에포 1000
# 로스는 :  0.617437481880188   / RMSE :  14.989367110154646 / 걸린시간 :  278.19 초  <- 위에 모델에서 에포 100
# 로스는 :  1.1099287271499634  / RMSE :  11.5352741007402   / 걸린시간 :  205.22 초  <- MinMaxScaler
# 로스는 :  0.66734778881073    / RMSE :  10.60895402061867  / 걸린시간 :  138.25 초  <- StandardScaler
# 로스는 :  0.7366823554039001  / RMSE :  1.7034246044033918 / 걸린시간 :  717.7 초  <- train_size=0.95 epochs=100
########################### MRSE 잘 못함.... #####################
# 로스는 :  1.4264005422592163 / RMSE :  1.7925319579656802  / 걸린시간 :  566.36 초 <- 심기일전
# 로스는 :  70.43020629882812  / RMSE :  12.390091410269848  / 걸린시간 :  130.87 초 <- 심기일전2
# 로스는 :  70.43251037597656  / RMSE :  12.417725529302217  / 걸린시간 :  122.96 초
# 로스는 :  70.42935180664062  / RMSE :  12.403575459578695  / 걸린시간 :  304.12 초 <- 진영님 모델 ㄱㄱ
# 로스는 :  0.46848756074905396 / RMSE :  1.4012267566686762 / 걸린시간 :  236.71 초 <- 진영님 모델 2트
# 로스는 :  0.8237671256065369 / RMSE :  2.0027935637159127  / 걸린시간 :  155.8 초  <- 진영님 모델 3트
# 로스는 :  0.4080936014652252 / RMSE :  2.0013564548835285  / 걸린시간 :  357.98 초 <- 진영님 모델 4트
# 로스는 :  0.3548697829246521 / RMSE :  1.6270212539006785  / 걸린시간 :  319.73 초 <- 진영님 모델 5트
# 로스는 :  0.27190613746643066 / RMSE :  1.507805557610548  / 걸린시간 :  686.79 초 <- 페이션 20, 에포 150
# 로스는 :  0.6138481497764587  / RMSE :  2.6234124142750543 / 걸린시간 :  430.46 초 <- 페이션 20, 에포 200
# 로스는 :  0.6138481497764587  / RMSE :  2.6234124142750543 / 걸린시간 :  430.46 초 <- 페이션 20, 에포 200 2

# 1등 0.7나옴.