from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import time
import numpy as np
import pandas as pd
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"    

#1. 데이터
path = 'C:/ai5/_data/kaggle/jena/'

csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)

train_dt = pd.DatetimeIndex(csv.index)

csv['day'] = train_dt.day
csv['month'] = train_dt.month
csv['year'] = train_dt.year
csv['hour'] = train_dt.hour
csv['dos'] = train_dt.dayofweek

y3 = csv.tail(144)
y3 = y3['T (degC)']

csv = csv[:-144]

x1 = csv.drop(['T (degC)', 'max. wv (m/s)', 'max. wv (m/s)', 'wd (deg)',"year"], axis=1)  # (420407, 13) <- T (degC) 없앰, 'wd (deg)'

y1 = csv['T (degC)']

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

x_test2 = x2[-1] # (144, 13, 1)

x_test2 = np.array(x_test2).reshape(1, 144, 15)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, 
                                                    shuffle= True,
                                                    random_state=3)

# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

# scaler = StandardScaler() # MinMaxSc aler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = np.reshape(x_train, (x_train.shape[0], 144, 13))
# x_test = np.reshape(x_test, (x_test.shape[0], 144, 13))

#2. 모델구성
#3. 컴파일, 훈련


#4. 평가, 예측
print("==================== 2. MCP 출력 =========================")
model = load_model('C:/ai5/_save/keras55/k55_jena0812_1617_0016-1.996190.hdf5')
results = model.evaluate(x_test, y_test) # results 결과 / evaluate 평가

y_pred = model.predict(x_test2) #batch_size=300

y_pred = np.array(y_pred).reshape(144, 1)  
##################################################################################
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse = RMSE(y3, y_pred)
###################################################################################

submit = pd.read_csv(path + "jena_climate_2009_2016.csv")

submit = submit[['Date Time','T (degC)']]
submit = submit.tail(144)
# print(submit)   # 

# y_submit = pd.DataFrame(y_pred)
# print(y_submit) # 

submit['T (degC)'] = y_pred
# print(submit)                  # [6493 rows x 1 columns]
# print(submit.shape)            # (6493, 1)

path2 = 'C:/ai5/_save/keras55'

submit.to_csv(path2 + "jena_김지혜.csv", index=False)

print("range(144)개의 결과 : ", y_pred)
print("로스는 : " , results[0])
print("RMSE : ", rmse)

# RMSE   1.4012267566686762
