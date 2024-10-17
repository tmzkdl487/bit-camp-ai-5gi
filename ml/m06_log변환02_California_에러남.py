from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
datasets = fetch_california_housing()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

# print(df) # [20640 rows x 8 columns]

df['target'] = datasets.target

# df.boxplot()    # df.plot.box()랑 똑같음. 
# df.plot.box() # 이건 선이 없이 나옴.
# plt.show()

# population 이거 이상해

# print(df.info())    
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 20640 entries, 0 to 20639
# Data columns (total 9 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   MedInc      20640 non-null  float64
#  1   HouseAge    20640 non-null  float64
#  2   AveRooms    20640 non-null  float64
#  3   AveBedrms   20640 non-null  float64
#  4   Population  20640 non-null  float64
#  5   AveOccup    20640 non-null  float64
#  6   Latitude    20640 non-null  float64
#  7   Longitude   20640 non-null  float64
#  8   target      20640 non-null  float64
# dtypes: float64(9)
# memory usage: 1.4 MB
# None 

# print(df.describe())

# df['Population'].boxplot()    # AttributeError: 'Series' object has no attribute 'boxplot' 시리즈에서 이러 안돼
# df['Population'].plot.box()   # 이거 돼.
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

############# x  Population 로그 변환 #################

# x['Population'] = np.log1p(x['Population']) # 지수변환 np.expm1
##############################################################

x_train, y_train, x_test, y_test = train_test_split(x, y, 
                                                    train_size=0.9, 
                                                    shuffle=True, 
                                                    random_state=42)  

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (18576, 8) (2064, 8) (18576,) (2064,)

# exit()

#################  y 로그 변환 #####################
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
#####################################################

# 2. 내 모델 
# model = Sequential()
# model.add(Dense(21, input_dim=8))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(15))
# model.add(Dense(10))
# model.add(Dense(1))

# 2. 선생님 모델
model = RandomForestRegressor(random_state=1234,
                              max_depth=5,   # 5가 디폴트
                              min_samples_split=3)  

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, ) # epochs=1000, batch_size=32

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)

score = model.score(x_test, y_test)
print('score : ', score)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

print("r2스코어 : ", r2)

# 디폴트 : 