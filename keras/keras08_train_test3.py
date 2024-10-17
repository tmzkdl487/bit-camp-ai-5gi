import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 나눠라
# 힌트: 사이킷런

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                        train_size=0.7, # 디폴트 0.75
                                        # test_size=0.3, (train_size든 test_size든 1개만 쓰면 된다. 0.4는 안되도 0.2는 된다.)
                                        # [검색2] train_size 디폴트 값 찾기.
                                        # train_size=0.75 가 디폴트 값이다.
                                        # shuffle=True, # 디폴트 True
                                        random_state=123,   # <- 데이터를 고정해준다.
                                        )     
print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_tast : ', y_test)
# def train_test_split(a, b):
#     a = a+b
#     return x_train, x_test, y_train, y_test

# x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.3) -> 현아님이 알려준 수식. 정답이였다.

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#.4 평가, 예측
print("++++++++++++++++++++++++++++++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)  
results = model.predict([11])
print("로스 : ", loss )
print('[11]의 예측값 : ', results) 

# 로스 :  1.1842379282265398e-15 [11]의 예측값 :  [[11.]]