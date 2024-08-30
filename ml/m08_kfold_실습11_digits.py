# m05_pca_evr_실습11_digits.py

from sklearn.datasets import load_digits

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터

x, y = load_digits(return_X_y=True)  # 데이터 다운하고 x, y 바로 됨.

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape)  # (1797, 10)

# print(x.shape, y.shape) # (1797, 64) (1797,)

# exit()

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
model = SVC(verbose=True)   # train_test_split 안써도 되는 모델임.

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print('digits_ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

# KFold 결과
# digits_ACC :  [0.98333333 0.98888889 0.98328691 0.99442897 0.98607242] 
#  평균 ACC :  0.9872