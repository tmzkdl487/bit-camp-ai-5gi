# keras41_ImageDataGernerator4_horse.py 복사

from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    # horizontal_flip=True,   # 증폭하면 이미지 망가질 수 있어서 주석처리 하라고 하심.
    # vertical_flip=True,    
    # width_shift_range=0.1, 
    # height_shift_range=0.1, 
    # rotation_range= 5,   
    # zoom_range=1.2,        
    # shear_range=0.7,       
    # fill_mode='nearest',   
)

test_datagen = ImageDataGenerator(
    rescale=1./255)

path_train = './_data/image/horse_human/'

start_time1 = time.time()

xy_train2 = train_datagen.flow_from_directory(
    path_train, 
    target_size=(10, 10), 
    batch_size=1027, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
) 

# print(xy_train2[0][0].shape, xy_train2[0][1].shape)  # (1027, 10, 10, 3) (1027,)

# exit()

xy_train = xy_train2[0][0].reshape(xy_train2[0][0].shape[0], xy_train2[0][0].shape[1]*xy_train2[0][0].shape[2]*xy_train2[0][0].shape[3])

# print(xy_train.shape) # (1027, 300)

# exit()

pca = PCA(n_components=300)  
xy_train = pca.fit_transform(xy_train)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 73
# print(np.argmax(cumsum >= 0.99) +1)  # 114
# print(np.argmax(cumsum >= 0.999) +1) # 188
# print(np.argmax(cumsum >= 1.0) +1)   # 1

# exit()

n = [73, 114, 188, 1]

y_train = to_categorical(xy_train2[0][1])

# 결과 저장
results = []

for i in range(0, len(n), 1):
    pca = PCA(n_components=n[i])
    x_train1 = pca.fit_transform(xy_train)
    
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

    model.fit(x_train1, xy_train2[0][1], epochs=10, batch_size=64, verbose=0, validation_split=0.2,
              callbacks=[es, mcp])
    
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_train1, xy_train2[0][1], verbose=0)
    
    print('===========================')
    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    
# ===========================
# 결과 PCA : 73
# acc :  0.9931840300559998
# 걸린 시간 :  4.0 초
# ===========================
# 결과 PCA : 114
# acc :  0.9912366271018982
# 걸린 시간 :  3.83 초
# ===========================
# 결과 PCA : 188
# acc :  0.9931840300559998
# 걸린 시간 :  4.0 초
# ===========================
# 결과 PCA : 1
# acc :  0.6173320412635803
# 걸린 시간 :  3.86 초    
