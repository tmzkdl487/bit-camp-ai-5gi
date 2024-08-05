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
test_datagen = ImageDataGenerator(
    rescale=1./255)

start_time = time.time()

np_path = 'c:/ai5/_data/_save_npy/'

x_train = np.load(np_path + 'keras45_horse_02_x_train.npy')
y_train = np.load(np_path + 'keras45_horse_02_y_train.npy')

end_time1 = time.time()
print("데이터 걸린시간 : ", round(end_time1 - start_time, 2), "초")

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=666)

end_time = time.time()

# 2. 모델
# 3. 컴파일, 훈련

#4. 평가, 예측
print("==================== 2. MCP 출력 =========================")
model = load_model('C:/ai5/_data/image/horse_human/k41_horse0805_1302_0030-0.020466.hdf5')
loss = model.evaluate(x_test, y_test, verbose=1)  

y_pred = model.predict(x_test)

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print(" 걸린시간 : ", round(end_time - start_time, 2), "초")

# 세이브한 가중치 값
# 로스는 :  0.0022340472787618637 / ACC :  1.0 / 데이터 걸린시간 :  4.92 초 / 걸린시간 :  11.34 초

# 확인한 값.
# 로스는 : 0.0 / ACC :  1.0 /  걸린시간 :  0.06 초
