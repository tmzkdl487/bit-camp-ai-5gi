# keras49_augment5_cat_dog.py 복사

# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
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

path = "./_data/kaggle/dogs-vs-cats-redux-kernels-edition/"
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

start_time1 = time.time()

np_path = 'c:/ai5/_data/_save_npy/'

# 개별 파일 불러오기
x_train_1 = np.load(np_path + 'keras43_01_x_train.npy')
x_train_2 = np.load(np_path + 'keras49_image_cat_dog_01_x_train.npy')
y_train_1 = np.load(np_path + 'keras43_01_y_train.npy')
y_train_2 = np.load(np_path + 'keras49_image_cat_dog_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

# 데이터를 합치기
x_train = np.concatenate((x_train_1, x_train_2), axis=0)
y_train = np.concatenate((y_train_1, y_train_2), axis=0)

# print(x_train.shape, y_train.shape) # (44997, 80, 80, 3) (44997,)
# print(x_test.shape, y_test.shape) # (12500, 80, 80, 3) (12500,)

# augment_size = 5000

# randidx = np.random.randint(x_train.shape[0], size=augment_size)    

# x_augmented = x_train[randidx].copy()   # .copy()하면 메모리값을 새로 할당하기 때문에 원래 메모리값에 영향을 미치지 않는다. 메모리 안전빵.
# y_augmented = y_train[randidx].copy()   #  x, y 5만개 준비됨.

# train_datagen = ImageDataGenerator(
#     rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
#     horizontal_flip=True,   # 수평 뒤집기
#     vertical_flip=True,     # 수직 뒤집기
#     width_shift_range=0.1,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
#     # height_shift_range=0.1, # 평행이동 수직
#     rotation_range= 2,      # 정해진 각도만큼 이미지 회전 
#     # zoom_range=1.2,         # 축소 또는 확대
#     # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
#     fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
# )

# # x_augmented = x_augmented.reshape(
# #     x_augmented.shape[0],         
# #     x_augmented.shape[1],          
# #     x_augmented.shape[2], 3) 

# # print(x_augmented.shape)    # (5000, 102, 102, 3)

# x_augmented = train_datagen.flow(
#     x_augmented, y_augmented,
#     batch_size=augment_size,
#     shuffle=False,
# ).next()[0]

# # x_train = x_train.reshape(5000, 102, 102, 3)
# # x_test = x_test.reshape(5000, 102, 102, 3)

# # print(x_train.shape, y_train.shape) # (5000,102,102,3)

# x_train = np.concatenate((x_train, x_augmented))   
# y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape)  # (30160, 80, 80, 3) (30160,)

xy_test=x_test

x_train = x_train/255.
x_test = x_test/255.

# print(x_train.shape, x_test.shape)  # (49997, 80, 80, 3) (12500, 80, 80, 3)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

# print(x_train.shape, x_test.shape)  # (44997, 19200) (12500, 19200)

# exit()

pca = PCA(n_components=19200)  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 679
# print(np.argmax(cumsum >= 0.99) +1)  # 3300
# print(np.argmax(cumsum >= 0.999) +1) # 7814
# print(np.argmax(cumsum >= 1.0) +1)   #  1

# exit()
n = [679, 3300, 7814, 1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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

    model.fit(x_train1, y_train, epochs=1, batch_size=64, verbose=0, validation_split=0.2,
              callbacks=[es, mcp])
    
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)
    
    print('===========================')
    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")

# ===========================
# 결과 PCA : 679
# acc :  0.27768000960350037
# 걸린 시간 :  2.61 초
# ===========================
# 결과 PCA : 3300
# acc :  0.6916000247001648
# 걸린 시간 :  3.04 초
# ===========================
# 결과 PCA : 7814
# acc :  0.42824000120162964
# 걸린 시간 :  3.73 초
# ===========================
# 결과 PCA : 1
# acc :  0.2791999876499176
# 걸린 시간 :  2.79 초