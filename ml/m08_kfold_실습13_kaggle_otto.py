# m05_pca_evr_실습13_kaggle_otto.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import xgboost as xgb

#1. 데이터
path = 'C://ai5//_data//kaggle//otto-group-product-classification-challenge//'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [61878 rows x 94 columns]
 
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
# print(test_csv)   # [144368 rows x 93 columns]
    
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(train_csv.shape, test_csv.shape, sampleSubmission_csv.shape)
# (61878, 94) (144368, 93) (144368, 9)

# [누리님 조언] 타겟을 숫자로 바꾼다.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
# print(x)    # [61878 rows x 93 columns]

y = train_csv['target']
# print(y.shape)  # (61878,)

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape) 

# print(x.shape, y.shape) # (61878, 93) (61878,)

# exit()

n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
# model = SVC(verbose=True)   # train_test_split 안써도 되는 모델임.
model = xgb.XGBClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print('kaggle_otto_ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

# KFold 결과
# kaggle_otto_ACC :  [0.81302521 0.81851972 0.80793471 0.81187879 0.81139394] 
#  평균 ACC :  0.8126