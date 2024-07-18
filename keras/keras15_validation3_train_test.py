import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터  
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

# [실습] 잘라라!!!
# train_test_split만 잘라라.

x_train = x[:13]
y_train = x[:13]

x_val = x[13:15]
y_val = x[13:15]

x_test = x[15:]
y_test = x[15:]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123) 
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.7, random_state=123)

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))    
model.add(Dense(10))   
model.add(Dense(1))    

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
result = model.predict([17])    

print("로스는 : ", loss)
print("[17]의 예측값", result)

