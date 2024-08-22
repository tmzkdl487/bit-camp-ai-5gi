from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target 

print(x)
print(y)
print(x.shape, y.shape) # (20640, 8) (20640,)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=3)

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=8))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print ("로스 : ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
               
# [실습] 맹그러 / R2 0.59 이상 -0.1 줄여주심.
# r2스코어 :  0.5343505878314927
# r2스코어 :  0.5153000248745865
# r2스코어 :  0.5407938909590657
# r2스코어 :  0.5532819039210624
# r2스코어 :  0.5520273955003091
# r2스코어 :  0.5884558602863272
