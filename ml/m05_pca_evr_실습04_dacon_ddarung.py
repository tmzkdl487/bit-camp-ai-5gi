# keras26_scaler04_dacon_ddarung.py

# https://dacon.io/competitions/open/235576/overview/description (대회 사이트 주소)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import pandas as pd
import sklearn as sk
import time

#1. 데이터
path = "C://ai5//_data//dacon//따릉이//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 인덱스 없으면 index_col쓰면 안됨. 0은 0번째 줄 없앴다는 뜻이다.
# print(train_csv)    # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
# print(submission_csv)   # [715 rows x 1 columns]

# print(train_csv.shape)  # (1459, 10)
# print(test_csv.shape) # (715, 9)
# print(submission_csv.shape) # (715, 1)

# print(train_csv.columns)    # 열의 이름을 알려달라는 수식. 
# # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
# #       dtype='object')

# print(train_csv.info()) # Non-Null이 몇갠지 구멍난 데이터가 있는지 확인하는 수식.

# ################# 결측치 처리 1. 삭제 ################### 행은 다 같아야 함.
# # print(train_csv.isnull().sum()) 밑에 코드도 같은 코드다. isnull이나 isna나 똑같다.
# print(train_csv.isna().sum())   # 구멍난 데이터의 수를 알려달라.   

train_csv = train_csv.dropna()  # 구멍난 데이터를 삭제해달라는 수식
# print(train_csv.isna().sum())   # 잘 지워졌는지 확인.

# print(train_csv)    # [1328 rows x 10 columns]
# print(train_csv.isna().sum())   # 다시 확인
# print(train_csv.info()) # 다시 확인

# print(test_csv.info())  # 이제 test 파일도 정보확인

test_csv = test_csv.fillna(test_csv.mean()) # 구멍난 데이터를 평균값으로 채워달라는 뜻.
# print(test_csv.info()) # 715 non-nul /  확인.

x = train_csv.drop(['count'], axis=1)   # train_csv에서 count 지우는 수식을 만들고 있다. count 컬럼의 axis는 가로 1줄을 지운다. 행을 지운다. []안해도 나온다.
# print(x)    # [1328 rows x 9 columns] / 확인해봄.
y = train_csv['count']  # y는 count 열만 가지고 옴. y를 만들고 있다.
# print(y.shape)  # (1328,)   # 확인해봄.

# print(x.shape, y.shape) # (1328, 9) (1328,)

# exit()

pca = PCA(n_components=8)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 1
# print(np.argmax(cumsum >= 0.99) +1)  # 1
# print(np.argmax(cumsum >= 0.999) +1) # 3
# print(np.argmax(cumsum >= 1.0) +1)   # 1

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    random_state=4343,
                                                    shuffle=True,
                                                    ) # random_state=3454, 맛집 레시피 : 4343 / stratify=y

n = [1, 3]

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
    model.compile(loss= 'mse', optimizer='adam', metrics=['acc'])

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

    model.fit(x_train1, y_train, epochs=10, batch_size=64, verbose=0, validation_split=0.2,
              callbacks=[es, mcp])
    
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)
    
    print('===========================')
    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")

# ===========================
# 결과 PCA : 1
# acc :  0.0
# 걸린 시간 :  1.62 초
# ===========================
# 결과 PCA : 3
# acc :  0.0
# 걸린 시간 :  1.0 초