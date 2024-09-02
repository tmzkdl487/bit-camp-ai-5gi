# m19_smote1_wine.선생님.py 복사

import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical

import tensorflow as tf
tf.random.set_seed(725)
    
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
# x = x[:-40]
# y = y[:-40]
# # print(y)    # <- 확인해봄. 2가 빠져서 불균형 데이터가 됨.
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]

# print(np.unique(y, return_counts=True)) # 다시 확인.
# (array([0, 1, 2]), array([59, 71,  8], dtype=int64)) <- 이렇게 나옴. 
# 전에는 (array([0, 1, 2]), array([59, 71, 48], dtype=int64))이랬음. 2가 28개 있었는데 8개로 만든 것임.

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.75, shuffle=True, random_state=333, stratify=y,)

'''
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

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
# results = model.score(x_test, y_test)
# print('model.score : ', results)

# # 지표: f1_score
# y_predict = model.predict(x_test)
# print('f1_score : ', f1_score(y_test, y_predict, average='macro'))

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('ACC : ', results[1])

# f1_score
y_predict = model.predict(x_test)

# print(y_predict)    # 3개의 값이 나오니까 argmax 해야겠지!!!!!

y_predict = np.argmax(y_predict, axis=1)
# print(y_predict) # [1 0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 2 1 1 0 1 0 1 0]

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average = 'macro')

print('acc : ', acc)
print('f1 : ', f1)

# loss :  0.7697742581367493
# ACC :  0.7714285850524902
# acc :  0.7714285714285715
# f1 :  0.5357142857142857

'''
x = x[:-39]
y = y[:-39]
# print(y)    # <- 확인해봄. 2가 빠져서 불균형 데이터가 됨.
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]

# print(np.unique(y, return_counts=True)) # 다시 확인.
# (array([0, 1, 2]), array([59, 71,  8], dtype=int64)) <- 이렇게 나옴. 
# 전에는 (array([0, 1, 2]), array([59, 71, 48], dtype=int64))이랬음. 2가 28개 있었는데 8개로 만든 것임.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=123, stratify=y,)

################# SMOTE 적용 ############################
# 근접한 데이터를 만들어줌.
# pip install imblearn

from imblearn.over_sampling import SMOTE

import sklearn as sk
# print('사이킷런 : ', sk.__version__)    # 사이킷런 :  1.1.3 / 선생님은 1.5.1

# print('증폭전 : ', np.unique(y_train, return_counts=True))

smote = SMOTE(random_state=7777)

x_train, y_train = smote.fit_resample(x_train, y_train)

# print(pd.value_counts(y_train))
# 0    53
# 1    53
# 2    53
# dtype: int64

# print('증폭후 : ', np.unique(y_train, return_counts=True))
# 증폭전 :  (array([0, 1, 2]), array([44, 53,  6], dtype=int64))
# 증폭후 :  (array([0, 1, 2]), array([53, 53, 53], dtype=int64))

################# SMOTE 적용 끝 ############################

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

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
# results = model.score(x_test, y_test)
# print('model.score : ', results)

# # 지표: f1_score
# y_predict = model.predict(x_test)
# print('f1_score : ', f1_score(y_test, y_predict, average='macro'))

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('ACC : ', results[1])

# f1_score
y_predict = model.predict(x_test)

# print(y_predict)    # 3개의 값이 나오니까 argmax 해야겠지!!!!!

y_predict = np.argmax(y_predict, axis=1)
# print(y_predict) # [1 0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 2 1 1 0 1 0 1 0]

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average = 'macro')

print('acc : ', acc)
print('f1 : ', f1)

# SMOTE 전
# loss :  0.7697742581367493
# ACC :  0.7714285850524902
# acc :  0.7714285714285715
# f1 :  0.5357142857142857

# SMOTE 후 -> 점수가 더 떨어짐. 다시 랜덤이랑 데이터 변경
# loss :  0.8302679657936096
# ACC :  0.7142857313156128
# acc :  0.7142857142857143
# f1 :  0.5110689437065149

# x = x[:-40] -> x = x[:-35] 변경
# y = y[:-40] -> y = y[:-35] 변경
# tf.random.set_seed(7777) -> 777로 변경
# loss :  0.5903050899505615
# ACC :  0.8611111044883728
# acc :  0.8611111111111112
# f1 :  0.7873015873015873

# tf.random.set_seed(725)로 변경
# loss :  0.7656592726707458
# ACC :  0.7777777910232544
# acc :  0.7777777777777778
# f1 :  0.6944444444444443

# SMOTE 후
# x = x[:-35] -> x = x[:-39] 변경
# y = y[:-35] -> y = y[:-39] 변경
# train_test_split에서 random_state=123로 변경
# loss :  0.19893686473369598
# ACC :  0.9714285731315613
# acc :  0.9714285714285714
# f1 :  0.9797235023041475