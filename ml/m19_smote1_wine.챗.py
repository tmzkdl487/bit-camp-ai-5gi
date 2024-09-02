import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

# print(x.shape, y.shape) # (178, 13) (178,)

# print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.value_counts(y))   # <- 판다스에서 컬럼 확인할때.
# 1    71
# 0    59
# 2    48
# dtype: int64

# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

# 불균형 데이터를 만들자.
x = x[:-40]
y = y[:-40]
# print(y)    # <- 확인해봄. 2가 빠져서 불균형 데이터가 됨.
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]

# print(np.unique(y, return_counts=True)) # 다시 확인.
# (array([0, 1, 2]), array([59, 71,  8], dtype=int64)) <- 이렇게 나옴. 
# 전에는 (array([0, 1, 2]), array([59, 71, 48], dtype=int64))이랬음. 2가 28개 있었는데 8개로 만든 것임.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=333, stratify=y,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
# model = XGBClassifier()

model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
# model.fit(x_train, y_train, 
#           eval_set = [x_test, y_test],  # <- 케라스에서 발리데이션 스플릿 데이터와 같음. 얼리 스탑쓸려면 필요함.
#           verbose=True,
#           )

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
# results = model.score(x_test, y_test)
# print('model.score : ', results)

# # 지표: f1_score
# y_predict = model.predict(x_test)
# print('f1_score : ', f1_score(y_test, y_predict, average='macro'))

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('ACC : ', result[1])

# f1_score
y_predict_prob = model.predict(x_test)

y_predict = np.argmax(y_predict_prob, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_labels, y_predict)
f1 = f1_score(y_test_labels, y_predict, average = 'macro')

print('acc : ', acc)
print('f1 : ', f1)
 
# loss :  0.8903377056121826
# ACC :  0.800000011920929
# acc :  0.8
# f1 :  0.5591591591591593