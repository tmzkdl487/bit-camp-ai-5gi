import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import sklearn as sk
print(sk.__version__)   # 0.24.2

from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']

'''
x = dataset.data
y = dataset.target

print(x)
print(x.shape)  # (506, 13)
print(y)  
print(y.shape)  # (506,)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=6666)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500000, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)
    
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2) 

# [과제] train_size = 0.7 ~ 0.9 사이 / r2 0.8 -0.1 줄여주심. 0.7 나오게 하기.

# train_size=0.75 r2스코어 :  0.7365823081353948
# train_size=0.7 / random_state=15645 / epochs=10000 / random_state=15645 / r2스코어 :  0.7007794834273509
# train_size=0.75 / random_state=8989 / epochs=1000 / batch_size=32 / r2스코어 :  0.7042565389216131
# train_size=0.7 / random_state=333333 /  epochs=3333/ batch_size=33 / r2스코어 :  0.7540095067146827/ r2스코어 :  0.760713613142604 / r2스코어 :  0.7463206127196489 / r2스코어 :  0.7573002851790058
# train_size=0.8 / random_state=333333 / epochs=3333 / batch_size=33 / r2스코어 :  0.779991349020035 / 0.7462829323349436 / r2스코어 :  0.7785035118528679 / r2스코어 :  0.7736757755732411
# r2스코어 :  0.741570838347922 / 0.7721948798609052 / r2스코어 :  0.77252773831965 / 0.7748238761491439
# r2스코어 :  0.7162032498512081 / r2스코어 :  0.7415556719594341 / r2스코어 :  0.7895978890121733
# r2스코어 :  0.789752025067346
'''