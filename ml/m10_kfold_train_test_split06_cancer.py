# m05_pca_evr_실습06_cancer.py

from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

import xgboost as xgb

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8)    # stratify=y

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
model = SVC(verbose=True)   # train_test_split 안써도 되는 모델임.

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold) 

y_predict = cross_val_predict(model, x_test, y_test)

acc = accuracy_score(y_test, y_predict)

print('cancer_ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
print('cross_val_predict ACC :', acc)


# KFold 결과
# ACC :  [0.92105263 0.87719298 0.90350877 0.94736842 0.91150442]
#  평균 ACC :  0.9121

# StratifiedKFold SVR 결과
# ACC :  [0.92105263 0.93859649 0.92105263 0.92982456 0.86725664]
#  평균 ACC :  0.9156

# train_test_split 결과
# cancer_ACC :  [0.95604396 0.95604396 0.96703297 1.         0.97802198]
#  평균 ACC :  0.9714
# cross_val_predict ACC : 0.956140350877193