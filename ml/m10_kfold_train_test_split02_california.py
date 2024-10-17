# m05_pca_evr_실습14_mnist.py

from sklearn.datasets import fetch_california_housing

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb


#1. 데이터

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8)    # stratify=y

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits=5  # 디폴트가 5/ 원래는 3이였음.
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
# model = SVR(verbose=True)   # train_test_split 안써도 되는 모델임.
# model = RandomForestRegressor()

model= xgb.XGBRFRegressor()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2') 
# print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
print('KFold R^2 Scores:', scores, '\n 평균 R^2:', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('Mean Squared Error:', mse)
print('cross_val_predict ACC :', r2)

# PCA 결과 ===========================
# 결과 PCA : 8
# acc :  0.0019379844889044762
# 걸린 시간 :  7.48 초 

# KFold SVR 결과
# ACC :  [-0.01169906 -0.02154014 -0.02692888 -0.02607676 -0.03588386] 
# 평균 ACC :  -0.0244

# train_test_split 결과
#  평균 R^2: 0.8018
# Mean Squared Error: 0.31843047367582183
# cross_val_predict ACC : 0.7605451303329237







