# 성우 하이텍 19일 월요일 종가를 맞춰봐!!!

# 제한시간 18일 일요일 23:59까지

# 앙상블 반드시 할 것!!!

# RNN 계열, 또는 Conv1D 쓸것!!!!

# 외부 데이터 사용 가능

# 외부 데이터 사용시 c:\ai5\_data\중간고사데이터\

from tensorflow.keras.models import Sequential, load_model  
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten,  Input, Concatenate , concatenate
from tensorflow.keras.layers import Bidirectional, Conv1D, MaxPool1D  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder

import time
import numpy as np
import pandas as pd

# 1. 데이터
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

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.95,
                                                    shuffle= True, random_state=3)

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

# 3. 컴파일, 훈련

# 4. 평가, 예측
print("==================== 2. MCP 출력 =========================")
model = load_model('C:/ai5/_save/중간고사가중치/keras63_99_성우하이텍_김지혜.hdf5')

loss = model.evaluate([x1_test, x2_test], y_test)

results = model.predict([x1_1test, x2_3test])

print("성우하이텍 8월19일 종가 : ", results[0][0])
print("로스는 : " , loss[0])

# 성우하이텍 8월19일 종가   7508.9375
