# 제출한 과제의 가중치를 불러와서 csv를 만들어서
# 선생님에게 검증한닷!!!

# 구라치면 손목 날라간다!!!

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

x_train = np.load(np_path + 'keras43_01_x_train.npy')
y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

xy_test=x_test

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.75, 
                                                    shuffle= True,
                                                    random_state=11)    # 83

end_time1 = time.time()

# print(xy_train[0][0].shape) # (25000, 200, 200, 3)

# 2. 모델
# 3. 컴파일, 훈련

#4. 평가, 예측
print("==================== 2. MCP 출력 =========================")
model = load_model('C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/k35_040804_2220_0019-0.630271.hdf5')
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=90)  

y_pred = model.predict(x_test, batch_size=90)

# xy_test = xy_test.to_numpy()
# xy_test = xy_test.reshape(200000, 10, 10, 2)

y_submit = model.predict(xy_test)

sample_submission_csv['label']= y_submit

# sample_submission_csv.to_csv(path + "sample_submission_kaggle_cat_dog_0805_0958.csv")
sample_submission_csv.to_csv(path + "sample_submission_kaggle_cat_dog_0805_1019.csv")

print("로스는 : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("데이터 걸린시간 : ", round(end_time1 - start_time1, 2), "초")
# print("걸린시간 : ", round(end_time2 - start_time2, 2), "초")

