# m05_pca_evr_실습11_digits.py

from sklearn.datasets import load_digits

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

#1. 데이터

x, y = load_digits(return_X_y=True)  # 데이터 다운하고 x, y 바로 됨.

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape)  # (1797, 10)

# print(x.shape, y.shape) # (1797, 64) (1797,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8)    # stratify=y

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
model = SVC(verbose=True)   # train_test_split 안써도 되는 모델임.

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold) 
print('digits_ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc)

# KFold 결과
# digits_ACC :  [0.98333333 0.98888889 0.98328691 0.99442897 0.98607242] 
#  평균 ACC :  0.9872

# train_test_split 결과