# 내가 한거보다 올려!!!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, Bidirectional
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

csv = csv[:-144]

# print(csv.shape)    # (420407, 14) <- 144개를 없앰. / (420407, 19)


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

# print(x2.shape)   # 
# print(y2.shape)   #

x = x2[:-1, :]
y = y2[1:]

# print(x.shape)  # (420263, 144, 13) -> x는 맨 뒤에를 날리고. / (420407, 17) / (420263, 144, 15)
# print(y.shape)  # (420263, 144)     -> y는 맨 앞을 날렸음.   / (420407,)    / (420263, 144)

# print(x2[-2])
# print(x[-1]) <- 잘 잘렸는지 확인해 봄.

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
model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(144, 15)))   # timesteps, features / activation='tanh'
model.add(Bidirectional(LSTM(120)))
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

################# Bidirectional 넣고 돌리기 ##########################
# 로스는 :  0.8076272010803223 / RMSE :  1.0672475583904284 / 걸린시간 :  398.99 초\
# 로스는 :  0.7007861137390137 / RMSE :  1.6500464027401207 / 걸린시간 :  1764.15 초