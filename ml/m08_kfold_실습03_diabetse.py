# m05_pca_evr_실습03_diabetse.py

from sklearn.datasets import load_diabetes

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
x, y = load_diabetes(return_X_y=True)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
model = SVC(verbose=True)   # train_test_split 안써도 되는 모델임.

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

# KFold 결과
# ACC :  [0.01123596 0.         0.         0.         0.01136364] 
# 평균 ACC :  0.0045