# 

from sklearn.datasets import fetch_california_housing

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor

#1. 데이터

x, y = fetch_california_housing(return_X_y=True)

n_splits=5  # 디폴트가 5/ 원래는 3이였음.
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
# model = SVR(verbose=True)   # train_test_split 안써도 되는 모델임.
model = RandomForestRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
# print(' KFold ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
print('StratifiedKFold_ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

# PCA 결과 ===========================
# 결과 PCA : 8
# acc :  0.0019379844889044762
# 걸린 시간 :  7.48 초 

# KFold SVR 결과
# ACC :  [-0.01169906 -0.02154014 -0.02692888 -0.02607676 -0.03588386] 
# 평균 ACC :  -0.0244

# StratifiedKFold SVR 결과
# ValueError: Supported target types are: ('binary', 'multiclass'). Got 'continuous' instead. 에러뜸





