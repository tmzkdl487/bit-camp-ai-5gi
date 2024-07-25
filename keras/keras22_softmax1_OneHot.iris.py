from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import tensorflow as tf
import numpy as np
import pandas as pd
import time

#1. 데이터
dataset = load_iris()
# print(dataset)

# print(dataset.DESCR)
# print(dataset.feature_names)    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) # (150, 4) (150,)

# print(y)    # 데이터가 순서대로 되어있어서 셔플 필수.
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

# print(np.unique(y, return_counts=True))    
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50

# from tensorflow.keras.utils import to_categorical # 누리님
# y = to_categorical(y)
# print(y)

# y = pd.DataFrame(y)
# y = pd.get_dummies(y)   # pd 이용 누리님
# print(y)

# from sklearn.preprocessing import OneHotEncoder #사이킷런활용 태운님
# oh = OneHotEncoder()
# y = pd.DataFrame(y)
# y = oh.fit_transform(y)
# print(y.shape)

# 맹그러봐!!!

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.9,
                                                    random_state= 1186,
                                                    stratify=y)

# print(pd.value_counts(y_trian))


# # 원 핫: 1. 케라스
# from tensorflow.keras.utils import to_categorical
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)  # (150, 3)

# 원 핫: 2. 판다스
y_ohe2 = pd.get_dummies(y)
print(y_ohe2.shape) # (150, 3)

# print("=============================================")
# # 원 핫: 3. 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# y_ohe3 = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)    # True가 디폴트
# y_ohe3 = ohe.fit_transform(y_ohe3)    # 아래 2개는 요거 1개를 2개로 나눈것.
# # ohe.fit(y_ohe3)
# # y_ohe3 = ohe.transform(y_ohe3)

# print(y_ohe3)


############# 맹그러봐 ###############

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe2, 
                                                    train_size=0.9,
                                                    random_state= 1186,
                                                    stratify=y)

#2. 모델
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights= True
)

model.fit(x_train, y_train, epochs=100, batch_size=8,
          verbose=1, validation_split=0.1, callbacks=[es])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

# y_pred = model.predict(x_test)
# print(y_pred[:20])
# y_pred = np.round(y_pred)
# print(y_pred[:20])

# accuracy_score = accuracy_score(y_test, y_pred)

print("로스는 : " , loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# acc_score 0.39473684210526316
# acc_score 0.6052631578947368

# [검색] 소프트 맥스를 찾아라. 

# 에큐러시 1.0 나오게 해라.
# ACC :  0.947
# ACC :  0.974
# ACC :  1.0

# [실습] stratify=y을 넣고 돌려보기.
# ACC :  1.0
