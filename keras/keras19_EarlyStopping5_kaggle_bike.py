# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv (카글 컴피티션 사이트)
# keras13_kaggle_bike1.py 수정

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
path = 'C:\\ai5\_data\\bike-sharing-demand\\'  

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)  # (10886, 11)
print(test_csv.shape)   # (6493, 10)
print(sampleSubmission.shape)   # (6493, 1)

########### x와 y를 분리
x  = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
print(x)    # [10886 rows x 10 columns]

y = train_csv['count']
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=33)

#2. 모델 구성   모래시계 모형은 안됨.
model = Sequential()
model.add(Dense(133, activation='relu', input_dim=8))   # relu는 음수는 무조껀 0으로 만들어 준다.
model.add(Dense(123, activation='relu'))
# model.add(Dense(113, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(93, activation='relu'))
# model.add(Dense(83, activation='relu'))
# model.add(Dense(73, activation='relu'))
# model.add(Dense(63, activation='relu'))
# model.add(Dense(53, activation='relu'))
# model.add(Dense(43, activation='relu'))
# model.add(Dense(33, activation='relu'))
# model.add(Dense(23, activation='relu'))
# model.add(Dense(13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear'))    # 원래는 linear이니 리니어를 친다.

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
start_time = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode ='min',
    patience = 10,
    restore_best_weights=True,
)

hist = model.fit(x_train, y_train, epochs=10, batch_size=32,
          verbose=1, 
          validation_split=0.3, )  
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)   # (10886,)

# sampleSubmission['count'] = y_submit
# print(sampleSubmission)
# print(sampleSubmission.shape)   #(10886,)

# sampleSubmission.to_csv(path + "sampleSubmission_0717_1523.csv")

print("로스는 : ", loss)
# print("cout의 예측값 : ", )
print("걸린시간 : ", round(end_time - start_time, 2), "초")

print("========================== hist ==============================")
print(hist)
print("======================= hist.histroy =========================")
print(hist.history)
print("============================ loss ============================")
print(hist.history['loss'])
print("======================= val_loss ============================")
print(hist.history['val_loss'])


from matplotlib import font_manager, rc # 폰트 세팅을 위한 모듈 추가
font_path = "C:/Windows/Fonts/malgun.ttf" # 사용할 폰트명 경로 삽입
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))  # 그림판 사이즈
plt.plot(hist.history['loss'], c ='red', label='loss')  #  marker='.'
plt.plot(hist.history['val_loss'], c ='blue', label='val_loss')
plt.legend(loc='upper right')   # 오른쪽 상단에 라벨값 써줌.
plt.rc('font', family = 'NanumGothic')
plt.title('카글 Kaggle Bike')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()