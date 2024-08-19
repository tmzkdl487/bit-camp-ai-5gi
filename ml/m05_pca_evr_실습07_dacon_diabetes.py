# keras26_scaler07_dacon_diabetes.py

# https://dacon.io/competitions/official/236068/overview/description

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
import time

#1. 데이터
path = 'C://ai5/_data/dacon/diabetes/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [652 rows x 9 columns]

test_csv = pd.read_csv(path + "test.csv", index_col = 0)
# print(test_csv) # [116 rows x 8 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape, test_csv.shape, sample_submission_csv.shape) 
# # (652, 9) (116, 8) (116, 1)

# print(train_csv.info())

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']

# print(x.shape, y.shape) # (652, 8) (652,)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
# exit()

pca = PCA(n_components=8)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 2
# print(np.argmax(cumsum >= 0.99) +1)  # 5
# print(np.argmax(cumsum >= 0.999) +1) # 6
# print(np.argmax(cumsum >= 1.0) +1)   # 8

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=455)

# print(x_train.shape, y_train.shape) # (65, 8) (65,) 
# print(x_test.shape, y_test.shape)   # (587, 8) (587,)

n = [2, 5, 6, 8]

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
    model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])

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
# 결과 PCA : 2
# acc :  0.6666666865348816
# 걸린 시간 :  1.35 초
# ===========================
# 결과 PCA : 5
# acc :  0.7424242496490479
# 걸린 시간 :  0.64 초
# ===========================
# 결과 PCA : 6
# acc :  0.7424242496490479
# 걸린 시간 :  0.64 초
# ===========================
# 결과 PCA : 8
# acc :  0.6212121248245239
# 걸린 시간 :  0.62 초

