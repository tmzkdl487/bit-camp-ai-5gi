# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import time

# # ## test 이미지 파일명 변경 ##
# import os
# import natsort

# file_path = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/test"
# file_names = natsort.natsorted(os.listdir(file_path))

# print(np.unique(file_names))
# i = 1
# for name in file_names:
#     src = os.path.join(file_path,name)
#     dst = str(i).zfill(5)+ '.jpg'
#     dst = os.path.join(file_path, dst)
#     os.rename(src, dst)
#     i += 1
    
# ###################

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    horizontal_flip=True,   # 증폭하면 이미지 망가질 수 있어서 주석처리 하라고 하심.
    vertical_flip=True,    
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    rotation_range= 10,   
    zoom_range=0.1,        
    shear_range=0.1,       
    fill_mode='nearest',   
)

test_datagen = ImageDataGenerator(
    rescale=1./255)

path_train = "./_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/"
path_test = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/"
path = "./_data/kaggle/dogs-vs-cats-redux-kernels-edition/"

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

start_time1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(102, 102), 
    batch_size=25000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)   # Found 25000 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(102, 102),  
    batch_size=12500,  
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
    # Found 12500 images belonging to 1 classes.
)  

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.75, 
                                                    shuffle= True,
                                                    random_state=11)    # 83

end_time1 = time.time()

# print(xy_train[0][0].shape) # (25000, 200, 200, 3)

xy_test=xy_test[0][0]

#2. 모델
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])   # acc넣어야 분류일 경우 잘 맞는지 확인할 수 있음.
start_time2 = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 5,
    restore_best_weights= True
)

########################### mcp 세이프 파일명 만들기 시작 ################
import datetime 
date = datetime.datetime.now()
print(date) # 2024-07-26 16:51:36.578483
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date) # 0726 / 0726_1654
print(type(date))

path = './_data/kaggle/dogs-vs-cats-redux-kernels-edition/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k35_04', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( 
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=1000, batch_size=64,
          validation_split=0.2, verbose=1, callbacks=[es, mcp])

end_time2 = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=90)  

y_pred = model.predict(x_test, batch_size=90)

# xy_test = xy_test.to_numpy()
# xy_test = xy_test.reshape(200000, 10, 10, 2)

y_submit = model.predict(xy_test)

sample_submission_csv['label']= y_submit

sample_submission_csv.to_csv(path + "sample_submission_kaggle_cat_dog_0805_0920.csv")

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("데이터 걸린시간 : ", round(end_time1 - start_time1, 2), "초")
print("걸린시간 : ", round(end_time2 - start_time2, 2), "초")

# 로스는 :  0.4071281850337982 /ACC :  0.807 /데이터 걸린시간 :  37.83 초 / 걸린시간 :  71.21 초 / 점수 1.26408
# 로스는 :  0.35915467143058777 / ACC :  0.842 / 데이터 걸린시간 :  54.04 초 / 걸린시간 :  175.55 초 / 점수 1.60550
# 로스는 :  0.4010143578052521 / ACC :  0.824 / 데이터 걸린시간 :  47.02 초 / 걸린시간 :  67.06 초 / 점수 1.45339
# 로스는 :  0.6932979822158813 / ACC :  0.489 / 데이터 걸린시간 :  46.78 초 / 걸린시간 :  35.21 초 / 점수 0.69316
# 로스는 :  0.6931857466697693 / ACC :  0.5 / 데이터 걸린시간 :  47.55 초 / 걸린시간 :  35.9 초 / 점수 0.69317
# 로스는 :  0.6931167840957642 / ACC :  0.507 / 데이터 걸린시간 :  46.99 초 / 걸린시간 :  101.51 초 / 점수 0.69314
# 로스는 :  0.6931564807891846 / ACC :  0.494 / 데이터 걸린시간 :  105.92 초 / 걸린시간 :  162.91 초 / 점수 0.69314
# 로스는 :  0.40134745836257935 / ACC :  0.833 / 데이터 걸린시간 :  51.43 초 / 걸린시간 :  376.07 초/ 점수 1.63167
# 로스는 :  0.6932631134986877 / ACC :  0.493 / 데이터 걸린시간 :  108.3 초 / 걸린시간 :  34.8 초 / 
