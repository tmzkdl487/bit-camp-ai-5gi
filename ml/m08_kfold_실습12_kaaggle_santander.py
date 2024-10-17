# m05_pca_evr_실습12_kaaggle_santander.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import xgboost as xgb

#1. 데이터
path = 'C://ai5/_data/kaggle/santander-customer-transaction-prediction/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [200000 rows x 201 columns]

test_csv = pd.read_csv(path + "test.csv", index_col = 0)
# print(test_csv) # [200000 rows x 200 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape, test_csv.shape, sample_submission_csv.shape)
# (200000, 201) (200000, 200) (200000, 1)

x  = train_csv.drop(['target'], axis=1) 
# print(x)    #[200000 rows x 200 columns]

y = train_csv['target']
# print(y.shape)  # (200000,)

n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
# model = SVC(verbose=True)   # train_test_split 안써도 되는 모델임.
model = xgb.XGBClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print('kaggle_santander_ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

# KFold 결과
# kaggle_santander_ACC :  [0.91375  0.91185  0.914425 0.910525 0.9143  ] 
#  평균 ACC :  0.913