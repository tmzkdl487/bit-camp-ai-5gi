# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

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

np_path = 'c:/ai5/_data/_save_npy/'

x_train_m = np.load(np_path + 'keras45_gender_05_Man_x_train.npy')
y_train_m = np.load(np_path + 'keras45_gender_05_Man_y_train.npy') 
x_train_w = np.load(np_path + 'keras45_gender_woman_05_x_train.npy')  # 
y_train_w = np.load(np_path + 'keras45_gender_woman_05_y_train.npy')  #  
x_test = np.load(np_path + 'keras45_gender_05_x_test.npy')  
y_test = np.load(np_path + 'keras45_gender_05_y_test.npy')  

# print(x_train_w.shape, y_train_w.shape) # (9489, 80, 80, 3) (9489,)
# print(x_train_m.shape, y_train_m.shape) # (17678, 80, 80, 3) (17678,)
# print(x_test.shape, y_test.shape) # (20000, 80, 80, 3) (20000,)

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.2,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range= 15,      # 정해진 각도만큼 이미지 회전 
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

augment_size = 9000

randidx = np.random.randint(x_train_w.shape[0], size=augment_size)    

x_augmented = x_train_w[randidx].copy()   # .copy()하면 메모리값을 새로 할당하기 때문에 원래 메모리값에 영향을 미치지 않는다. 메모리 안전빵.
y_augmented = y_train_w[randidx].copy()

# print(x_augmented.shape)    # (9000, 80, 80, 3)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],         
    x_augmented.shape[1],          
    x_augmented.shape[2], 3) 

# print(x_augmented.shape)    # (9000, 80, 80, 3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

x_train_w = np.concatenate((x_train_w, x_augmented))
# print(x_train_w.shape)  # (18489, 80, 80, 3)
   
y_train_w = np.concatenate((y_train_w, y_augmented))   
# print(y_train_w.shape)  # (18489,)

x = np.concatenate((x_train_m, x_train_w))
y = np.concatenate((y_train_m, y_train_w))

print(x.shape, y.shape) # (36167, 80, 80, 3) (36167,)

np_path2='C:/ai5/_data/image/me/'
x_test2=np.load(np_path2 + 'me3.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    shuffle= True,  random_state=11)   

#2. 모델
model = Sequential()
model.add(Conv2D(32, 2, activation='relu', input_shape = (80, 80, 3), padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(128, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation='relu', padding='same'))
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
    patience = 10,
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

path = './_save/keras35/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'
filepath = "".join([path, 'k49_gender', date, '_', filename])
# 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

########################### mcp 세이프 파일명 만들기 끗 ################

mcp = ModelCheckpoint( 
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_olny=True, 
    filepath = filepath,
)

model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.2, verbose=1, callbacks=[es, mcp])

end_time2 = time.time()

#4. 평가, 예측
# print("==================== 2. MCP 출력 =========================")
# model = load_model('C:/ai5/_data/image/brain/k41_brain0805_1223_0081-0.014112.hdf5')
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16)  

y_pred = model.predict(x_test2, batch_size=16)

y_pred = np.clip(y_pred, 1e-6, 1-(1e-6))


print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
# print(" 데이터 걸린시간 : ", round(end_time1 - start_time1, 2), "초")
print(" 걸린시간 : ", round(end_time2 - start_time2, 2), "초")
print(y_pred)   # 내가 남자지 여잔지 결과

# 로스는 :  0.3053964376449585 / ACC :  0.86 /  걸린시간 :  387.9 초

# 이미지 확인 후
# 로스는 :  0.27371418476104736 / ACC :  0.875 /  걸린시간 :  396.01 초
# 로스는 :  0.2773433029651642 / ACC :  0.876 /  걸린시간 :  396.59 초/ [[0.]]

# y_pred = np.clip(y_pred, 1e-6, 1-(1e-6)) 현아님이 알려준 이 코드 넣으니까 여자나옴!!!!
# 로스는 :  0.26245108246803284 / ACC :  0.88 /  걸린시간 :  514.23 초 / [[1.e-06]]


# ################# 내가 남자인지 여자인지 확인. 현아님이 수정해줘서 이렇게 할 필요 없음. 
# 그냥 바로 테스트에 내 사진을 넣고 돌렸음.

# path = 'C:/ai5/_data/image/me/'

# x_test = np.load(path + 'me3.npy')

# model = load_model('C:/ai5/_data/kaggle/biggest_gender/k49_gender0806_2057_0034-0.327698.hdf5')

# y_predict = model.predict(x_test)

# print(np.round(y_predict))

# # [[0.]] <= 남자로 나옴.