# m05_pca_evr_실습06_cancer.py

from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
model = SVC(verbose=True)   # train_test_split 안써도 되는 모델임.

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))


# KFold 결과
# ACC :  [0.92105263 0.87719298 0.90350877 0.94736842 0.91150442]
#  평균 ACC :  0.9121

# StratifiedKFold SVR 결과
# ACC :  [0.92105263 0.93859649 0.92105263 0.92982456 0.86725664]
#  평균 ACC :  0.9156