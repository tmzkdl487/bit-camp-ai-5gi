from sklearn.datasets import load_iris

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8, stratify=y,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
model = SVC()   # train_test_split 안써도 되는 모델임.

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)  # fit 대신 모델의 정확도를 보여줌.
print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
# ACC :  [0.95833333 0.95833333 0.95833333 1.         1.        ] 
#  평균 ACC :  0.975

y_predict = cross_val_predict(model, x_test, y_test)

# print(y_predict)
# print(y_test)
# [1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 1 1 2 2 1 1 1 1 1]
# [1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 2 2 1 2 2 1 1 1 1]

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc)   
# cross_val_predict ACC : 0.8666666666666667 <- 이렇게 점수가 나쁜 이유. 데이터가 작아서.