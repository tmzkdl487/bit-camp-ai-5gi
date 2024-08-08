# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

# y는 T(degC)로 잡아라. 144개

# 자르는 거는 맘대로

# 31.12.2016 00:10:00 부터
# 01.01.2017 00:00:00 까지

# 맞춰라!!! 1, 2, 3등 상 줌. 일요일 12시 59분까지

# LMSE로

# y의 shape는 (n, 144)
# 프레딕은 (1, 144)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU , LSTM 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd

#1. 데이터
path = 'C:/ai5/_data/kaggle/jena/'

csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)

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

# print(x1.shape) # (420407, 13)
# print(y1.shape) # (420407,)

size = 48

def split_x(dataset, size): 
    aaa = []
    for i in range(len(dataset) - size + 1):   
        subset = dataset[i : (i + size)]
        aaa.append(subset)                 
    return np.array(aaa)

x2 = split_x(x1, size)  

# print(x2.shape)   # (420360, 48, 13)

y2 = split_x(y1, size)

# print(y2.shape) # (420360, 48)

x = x2[:, :-1] 
y = y2[:, :]

x_test2 = x[:-1]

# print(x.shape)  # (420360, 47, 13)
# print(y.shape)  # (420360, 48) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    shuffle= True,
                                                    random_state=666)

# 2. 모델
model = Sequential()
model.add(LSTM(units=30, activation='tanh', input_shape=(47, 13))) # timesteps, features
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(144))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    patience = 5,
    restore_best_weights= True
)

model.fit(x, y, epochs=1000, batch_size=64, verbose=1, callbacks=[es])  # validation_split=0.1, mcp

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss: ', results)

x_test2 = np.array(x1).reshape(1, 47, 13)   
y_pred = model.predict(x_test2)

print("range(144)개의 결과 : ", y_pred)
