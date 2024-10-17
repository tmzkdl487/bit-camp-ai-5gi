# keras61_DNN_jena.py

# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.decomposition import PCA
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

x = x2[:-1, :]
y = y2[1:]

############## DNN으로 바꾸기

x = x.reshape(420263, 144*15)

# print(x.shape, y.shape) # (420263, 2160) (420263, 144)

# exit()

x_test2 = x2[-1] 

# x_test2 = x_test2.reshape(1, 144*15)

# print(x_test2.shape)    # (144, 15)

# exit()

x_test2 = np.array(x_test2).reshape(1, 144*15)

# print(x_test2.shape)    # (1, 144, 13) / (1, 144, 17)

# exit()

pca = PCA(n_components=2160)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 33
# print(np.argmax(cumsum >= 0.99) +1)  # 108
# print(np.argmax(cumsum >= 0.999) +1) # 215
# print(np.argmax(cumsum >= 1.0) +1)   # 2160

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    random_state=4343,
                                                    shuffle=True,
                                                    )


n = [33, 108, 215, 2160]

# 결과 저장
results = []

for i in range(0, len(n), 1):
    pca = PCA(n_components=n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=n[i]))   # relu는 음수는 무조껀 0으로 만들어 준다.
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))   # activation='softmax'
    
    #3. 컴파일, 훈련
    model.compile(loss= 'mse', optimizer='adam', metrics=['acc'])   # 'binary_crossentropy' , 'categorical_crossentropy'

    start = time.time()
    
    es = EarlyStopping(monitor='val_loss', mode='min',
                    patience=5, verbose=0,
                    restore_best_weights=True)
    
    ########################### mcp 세이프 파일명 만들기 시작 ################
    import datetime 
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/m05/'
    filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'    
    filepath = "".join([path, 'm04_03_date_', str(i+1),'_', date, '_epo_', filename])
    ########################### mcp 세이프 파일명 만들기 끗 ################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    model.fit(x_train1, y_train, epochs=1, batch_size=64, verbose=0, validation_split=0.2,
              callbacks=[es, mcp])
    
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)
    
    print('===========================')
    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    
# ===========================
# 결과 PCA : 33
# acc :  0.000434412359027192
# 걸린 시간 :  21.95 초
# ===========================
# 결과 PCA : 108
# acc :  0.0004912542644888163
# 걸린 시간 :  24.52 초
# ===========================
# 결과 PCA : 215
# acc :  0.0004679557168856263
# 걸린 시간 :  24.99 초
# ===========================
# 결과 PCA : 2160
# acc :  0.00035162773565389216
# 걸린 시간 :  31.54 초