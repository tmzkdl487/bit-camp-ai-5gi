# m05_pca_evr_실습21_jena.py

# keras61_DNN_jena.py

# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam  

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import time

import random as rn
import tensorflow as tf

tf.random.set_seed(337)
np.random.seed(337)
rn.seed(337)

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

y3 = csv.tail(144)['T (degC)']

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

x = x2[:-1]
y = y2[1:]

############## DNN으로 바꾸기

x = x.reshape(420263, 144*15)

# print(x.shape, y.shape) # (420263, 2160) (420263, 144)

# exit()

x_test2 = x2[-1] 

x_test2 = x_test2.reshape(1, 144*15)

# print(x_test2.shape)    # (144, 15)

# exit()

x_test2 = np.array(x_test2).reshape(1, 144*15)

# print(x_test2.shape)    # (1, 144, 13) / (1, 144, 17)

# exit()

# print(x.shape, y.shape) # (420263, 2160) (420263, 144)

# exit()

pca = PCA(n_components=33)  
x = pca.fit_transform(x)

# print(x.shape, y.shape) # (420263, 33) (420263, 144)

# exit()


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, 
                                                    random_state=4343,
                                                    shuffle=True,
                                                    )

# print(x_train.shape, y_train.shape) # (315197, 33) (315197, 144)
# print(x_test.shape, y_test.shape)   # (105066, 33) (105066, 144)
# exit()

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

# 결과 저장
results = []

for learning_rate in lr:

    #2. 모델
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=x_train.shape[1]))   # relu는 음수는 무조껀 0으로 만들어 준다.
    model.add(Dense(10))
    model.add(Dense(144)) # , activation='softmax'
    
    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate)) # 'categorical_crossentropy'

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1,
          batch_size=32, 
          verbose=0
          )

    #4. 평가, 예측
    print("======================= 1. 기본출력 =============================")

    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr: {0}, 로스:{1}'.format(learning_rate, loss))

    y_predict = model.predict(x_test, verbose=0)
    
    r2 = r2_score(y_test,  y_predict )
    print('lr: {0}, r2: {1}'.format(learning_rate, r2))

# ======================= 1. 기본출력 =============================
# lr: 0.1, 로스:2324496128.0
# lr: 0.1, r2: -32750146.248233214
# ======================= 1. 기본출력 =============================
# lr: 0.01, 로스:2884.74853515625
# lr: 0.01, r2: -39.6440251598762
# ======================= 1. 기본출력 =============================
# lr: 0.005, 로스:45.468467712402344
# lr: 0.005, r2: 0.35935510689335565
# ======================= 1. 기본출력 =============================
# lr: 0.001, 로스:25.646533966064453
# lr: 0.001, r2: 0.6386520647140831
# ======================= 1. 기본출력 =============================
# lr: 0.0005, 로스:33.455196380615234
# lr: 0.0005, r2: 0.5286164352475423
# ======================= 1. 기본출력 =============================
# lr: 0.0001, 로스:463.5735778808594
# lr: 0.0001, r2: -5.5317804133997575
