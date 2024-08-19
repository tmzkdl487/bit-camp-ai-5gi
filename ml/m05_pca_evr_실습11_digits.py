# keras26_scaler11_digits.py
# keras22_softmax4_digits.py 복사

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터

x, y = load_digits(return_X_y=True)  # 데이터 다운하고 x, y 바로 됨.

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape)  # (1797, 10)

# print(x.shape, y.shape) # (1797, 64) (1797,)

# exit()

pca = PCA(n_components=64)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 29
# print(np.argmax(cumsum >= 0.99) +1)  # 41
# print(np.argmax(cumsum >= 0.999) +1) # 49
# print(np.argmax(cumsum >= 1.0) +1)   # 61

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    random_state=4343,
                                                    shuffle=True,
                                                    )

# print(x_train.shape, y_train.shape) # (455, 13) (455,)
# print(x_test.shape, y_test.shape)   # (51, 13) (51,)

n = [29, 41, 49, 61]

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
# 결과 PCA : 29
# acc :  0.20555555820465088
# 걸린 시간 :  1.78 초
# ===========================
# 결과 PCA : 41
# acc :  0.20555555820465088
# 걸린 시간 :  1.16 초
# ===========================
# 결과 PCA : 49
# acc :  0.20000000298023224
# 걸린 시간 :  1.18 초
# ===========================
# 결과 PCA : 61
# acc :  0.20555555820465088
# 걸린 시간 :  1.17 초
