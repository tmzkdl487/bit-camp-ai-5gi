# m35_원래m03_xor.py 카피

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])
print(x_data.shape, y_data.shape)   # (4, 2) (4,)

#2. 모델
# model = LinearSVCR()
# model = Perceptron()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_data, y_data, batch_size=12, epochs=100)

#4. 평가, 예측
# acc = model.score(x_data, y_data)
# print('model.score : ', acc)

loss = model.evaluate(x_data, y_data)
print('acc : ', loss[1])

y_predict = model.predict(x_data)
y_predict = np.round(y_predict).reshape(-1,).astype(int)

acc2 = accuracy_score(y_data, y_predict)
print('accuracy_score : ', acc2)

print("================================")
print(y_data)
print(y_predict)

# acc :  0.5
# accuracy_score :  0.5
# ================================
# [0 1 1 0]
# [0 1 0 1]

# 모델 건들지 말고 acc 1.0 만들 것. <- 해결이 안됨.