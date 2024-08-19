# keras26_scaler13_kaggle_otto.py

# keras23_kaggle2_otto.py 복사

# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터
path = 'C://ai5//_data//kaggle//otto-group-product-classification-challenge//'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [61878 rows x 94 columns]
 
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
# print(test_csv)   # [144368 rows x 93 columns]
    
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(train_csv.shape, test_csv.shape, sampleSubmission_csv.shape)
# (61878, 94) (144368, 93) (144368, 9)

# [누리님 조언] 타겟을 숫자로 바꾼다.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
# print(x)    # [61878 rows x 93 columns]

y = train_csv['target']
# print(y.shape)  # (61878,)

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape) 

# print(x.shape, y.shape) # (61878, 93) (61878,)

# exit()

pca = PCA(n_components=13)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 1
# print(np.argmax(cumsum >= 0.99) +1)  # 1
# print(np.argmax(cumsum >= 0.999) +1) # 1
# print(np.argmax(cumsum >= 1.0) +1)   # 1

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    random_state=4343,
                                                    shuffle=True,
                                                    )

# print(x_train.shape, y_train.shape) # (455, 13) (455,)
# print(x_test.shape, y_test.shape)   # (51, 13) (51,)

n = [1]

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
# acc :  0.25840336084365845
# 걸린 시간 :  22.68 초