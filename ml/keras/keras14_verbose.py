# keras14_verbose.py 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터  
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])
y_val = np.array([7,8])

x_test = np.array([9,10])
y_test = np.array([9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, 
          verbose=1) 
# verbose=0은 === 안 나옴./ 1은 디폴트 / 2를 넣으면 =====없음./ ~ 3은 에포만 나옴
# verbose=0 : 침묵
# verbose=1 : 디폴트
# verbose=2 : 프로그래스바 삭제
# verbose=나머지 : 에포만 나온다. 

#.4 평가, 예측
print("++++++++++++++++++++++++++++++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)   # 데이터는 2개로 나눌 것이다.
results = model.predict([11])
print("로스 : ", loss )
print('[11]의 예측값 : ', results)

