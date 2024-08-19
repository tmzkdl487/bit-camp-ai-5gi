# keras38_dnn1_mnist.py

# keras35_cnn4_mnist.py 카피

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

# print(x_train.shape, x_test.shape) #  (60000, 28, 28) (10000, 28, 28)

# exit()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

# print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

# exit()

pca = PCA(n_components=784)  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 154
# print(np.argmax(cumsum >= 0.99) +1)  # 331
# print(np.argmax(cumsum >= 0.999) +1) # 486
# print(np.argmax(cumsum >= 1.0) +1)   # 1

# exit()

n = [154, 331, 486, 1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 결과 저장
results = []

for i in range(0, len(n), 1):
    pca = PCA(n_components=n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    # print(x_train1.shape, x_test1.shape) # (60000, 154) (10000, 154)
    
    # exit()
    
    #2. 모델
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=n[i]))   # relu는 음수는 무조껀 0으로 만들어 준다.
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))   # activation='softmax'
    
    #3. 컴파일, 훈련
    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])   # 'binary_crossentropy' , 'categorical_crossentropy'

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
# 결과 PCA : 154
# acc :  0.9714999794960022
# 걸린 시간 :  16.01 초
# ===========================
# 결과 PCA : 331
# acc :  0.9718000292778015
# 걸린 시간 :  23.0 초
# ===========================
# 결과 PCA : 486
# acc :  0.9728999733924866
# 걸린 시간 :  19.46 초
# ===========================
# 결과 PCA : 1
# acc :  0.3098999857902527
# 걸린 시간 :  23.65 초

    
    