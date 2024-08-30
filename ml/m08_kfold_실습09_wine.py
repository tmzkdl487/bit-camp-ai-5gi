# m05_pca_evr_실습09_wine.py

from sklearn.datasets import load_wine

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
dataset = load_wine()

x = dataset.data
y = dataset.target
# print(x.shape, y.shape) # (178, 13) (178,)

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape)  # (178, 3)

# print(x.shape, y.shape) # (178, 13) (178,)

# exit()

n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
model = SVC(verbose=True)   # train_test_split 안써도 되는 모델임.

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print('Wine_ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

# KFold 결과
# Wine_ACC :  [0.69444444 0.63888889 0.61111111 0.74285714 0.62857143]
#  평균 ACC :  0.6632