# keras25_input_shape.py 복사.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import time

#1. 데이터
dataset = load_boston()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data
y = dataset.target

# print(x)
# print(x.shape)  # (506, 13)
# print(y)  
# print(y.shape)  # (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=6666)

#2. 모델구성
model = Sequential()
# model.add(Dense(1, input_dim=13))
model.add(Dense(1, input_shqpe=(13,)))    # 이미지 input_shape=(8, 8, 1)
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()


hist = model.fit(x_train, y_train, epochs=2000, batch_size=16,   # hist는 히스토리를 줄인말이다.
          verbose=1, validation_split=0.3,
          callbacks = [es]  #얼리스타핑을 콜백한다.
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)
    
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2) 
print("걸린시간 : ", round(end_time - start_time, 2), "초")

print("========================== hist ==============================")
print(hist)
print("======================= hist.histroy =========================")
print(hist.history)
print("============================ loss ============================")
print(hist.history['loss'])
print("======================= val_loss ============================")
print(hist.history['val_loss'])

# [실습] 파이썬 딕셔너리 검색

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))  # 그림판 사이즈
plt.plot(hist.history['loss'], c ='red', label='loss')  #  marker='.'
plt.plot(hist.history['val_loss'], c ='blue', label='val_loss')
plt.legend(loc='upper right')   # 오른쪽 상단에 라벨값 써줌.
plt.title('Boston Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

# [과제] train_size = 0.7 ~ 0.9 사이 / r2 0.8 -0.1 줄여주심. 0.7 나오게 하기.

# train_size=0.75 r2스코어 :  0.7365823081353948
# train_size=0.7 / random_state=15645 / epochs=10000 / random_state=15645 / r2스코어 :  0.7007794834273509
# train_size=0.75 / random_state=8989 / epochs=1000 / batch_size=32 / r2스코어 :  0.7042565389216131
# train_size=0.7 / random_state=333333 /  epochs=3333/ batch_size=33 / r2스코어 :  0.7540095067146827/ r2스코어 :  0.760713613142604 / r2스코어 :  0.7463206127196489 / r2스코어 :  0.7573002851790058
# train_size=0.8 / random_state=333333 / epochs=3333 / batch_size=33 / r2스코어 :  0.779991349020035 / 0.7462829323349436 / r2스코어 :  0.7785035118528679 / r2스코어 :  0.7736757755732411
# r2스코어 :  0.741570838347922 / 0.7721948798609052 / r2스코어 :  0.77252773831965 / 0.7748238761491439
# r2스코어 :  0.7162032498512081 / r2스코어 :  0.7415556719594341 / r2스코어 :  0.7895978890121733
# r2스코어 :  0.789752025067346

# verbose=0, validation_split=0.3을 넣어서 다시 해봄.
# r2스코어 :  0.8318105597373147 / 
# validation_split=0.4을 넣어서 다시 해봄.
# r2스코어 :  0.8390326228175955 / r2스코어 :  0.856450444072601