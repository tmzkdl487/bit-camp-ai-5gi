# m05_pca_evr_실습05_kaggle_bike.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

#1. 데이터
path = 'C://ai5/_data/kaggle//bike-sharing-demand/'  

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)  # (10886, 11)
# print(test_csv.shape)   # (6493, 10)
# print(sampleSubmission.shape)   # (6493, 1)

########### x와 y를 분리
x  = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
# print(x)    # [10886 rows x 10 columns]

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8)    # stratify=y

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델
model = SVC(verbose=True)   # train_test_split 안써도 되는 모델임.

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold) 

y_predict = cross_val_predict(model, x_test, y_test)

acc = accuracy_score(y_test, y_predict)

print('kaggle_bike_ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
print('cross_val_predict ACC :', acc)

# KFold 결과
# ACC :  [0.01698806 0.01607717 0.01561782 0.01332108 0.01791456] 
#  평균 ACC :  0.016

# train_test_split 결과
# cross_val_predict ACC : 0.021120293847566574