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

augment_size = 5000

randidx = np.random.randint(x_train.shape[0], size=augment_size)    

x_augmented = x_train[randidx].copy()   # .copy()하면 메모리값을 새로 할당하기 때문에 원래 메모리값에 영향을 미치지 않는다. 메모리 안전빵.
y_augmented = y_train[randidx].copy()   #  x, y 5만개 준비됨.

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range= 2,      # 정해진 각도만큼 이미지 회전 
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

# x_augmented = x_augmented.reshape(
#     x_augmented.shape[0],         
#     x_augmented.shape[1],          
#     x_augmented.shape[2], 3) 

# print(x_augmented.shape)    # (5000, 102, 102, 3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

# x_train = x_train.reshape(5000, 102, 102, 3)
# x_test = x_test.reshape(5000, 102, 102, 3)

# print(x_train.shape, y_train.shape) # (5000,102,102,3)

x_train = np.concatenate((x_train, x_augmented))   
y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape)  # (30160, 80, 80, 3) (30160,)

xy_test=x_test

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.75, 
                                                    shuffle= True,
                                                    random_state=11)    # 83

end_time1 = time.time()

#2. 모델
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(80, 80, 3), padding='same')) 
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
# print(date) # 2024-07-26 16:51:36.578483
# print(type(date))
date = date.strftime("%m%d_%H%M")
# print(date) # 0726 / 0726_1654
# print(type(date))

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

model.fit(x_train, y_train, epochs=10, batch_size=2,
          validation_split=0.2, verbose=1, callbacks=[es, mcp])

end_time2 = time.time()

#4. 평가, 예측
# print("==================== 2. MCP 출력 =========================")
# model = load_model('C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/k35_040804_2220_0019-0.630271.hdf5')
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16)  

y_pred = model.predict(x_test, batch_size=16)

# xy_test = xy_test.to_numpy()
# xy_test = xy_test.reshape(200000, 10, 10, 2)

y_submit = model.predict(xy_test)

sample_submission_csv['label']= y_submit

# sample_submission_csv.to_csv(path + "sample_submission_kaggle_cat_dog_0805_0958.csv")
sample_submission_csv.to_csv(path + "sample_submission_kaggle_cat_dog_0806_1600.csv")

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("데이터 걸린시간 : ", round(end_time1 - start_time1, 2), "초")
# print("걸린시간 : ", round(end_time2 - start_time2, 2), "초")

# 로스는 :  0.6934699416160583 / ACC :  0.498 / 데이터 걸린시간 :  6.22 초

# 이미지 확인 후 돌린 것
# 로스는 :  0.6933377385139465 / ACC :  0.499 / 데이터 걸린시간 :  6.95 초
