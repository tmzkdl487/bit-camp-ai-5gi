
from sklearn.datasets import load_iris

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
# x, y = load_boston(return_X_y=True)
# print(x)
# [[6.3200e-03 1.8000e+01 2.3100e+00 ... 1.5300e+01 3.9690e+02 4.9800e+00]
#  [2.7310e-02 0.0000e+00 7.0700e+00 ... 1.7800e+01 3.9690e+02 9.1400e+00]
#  [2.7290e-02 0.0000e+00 7.0700e+00 ... 1.7800e+01 3.9283e+02 4.0300e+00]
#  ...
#  [6.0760e-02 0.0000e+00 1.1930e+01 ... 2.1000e+01 3.9690e+02 5.6400e+00]
#  [1.0959e-01 0.0000e+00 1.1930e+01 ... 2.1000e+01 3.9345e+02 6.4800e+00]
#  [4.7410e-02 0.0000e+00 1.1930e+01 ... 2.1000e+01 3.9690e+02 7.8800e+00]]

x, y = load_iris(return_X_y=True)

# print(x.shape, y.shape) # (150, 4) (150,)

n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
model = SVC()   # train_test_split 안써도 되는 모델임.

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold) # cv=kfold에서 cv은 교차 검증 전략
print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

# print('ACC : ', scores)으로 돌리면 밑에처럼 나오고
# ACC :  [1.         0.86666667 1.         0.96666667 0.96666667]

# print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))로 바꾸면 밑에처럼 나옴.
# ACC :  [1.         0.86666667 1.         0.96666667 0.96666667] 
#  평균 ACC :  0.96

# StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)
# ACC :  [0.93333333 0.96666667 0.93333333 1.         1.        ] 
#  평균 ACC :  0.9667





