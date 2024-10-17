import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                        train_size=0.7, 
                                        random_state=1004, 
                                        )     
print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_tast : ', y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense (3))
model.add(Dense (3))
model.add(Dense (3))
model.add(Dense (1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#.4 평가, 예측
loss = model.evaluate(x_test, y_test)  
results = model.predict(x)
print("로스 : ", loss )
print('[11]의 예측값 : ', results) 

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x,results, color='red')
plt.show()

# 로스 :  2.3054082703310996e-05 [11]의 예측값 :  [[10.995116]]
# 로스 :  0.0027625737711787224 [11]의 예측값 :  [[10.896573]]