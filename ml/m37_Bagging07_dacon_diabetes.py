import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression # <- 분류 모델
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

import warnings
warnings.filterwarnings('ignore')

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

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    # stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()
model = BaggingClassifier(DecisionTreeClassifier(),
                          n_estimators =100,
                          n_jobs = -1,
                          random_state=4444, 
                        #   bootstrap=True,   # 디폴트, 중복 허용
                          bootstrap=False     # 중복허용 안함.
                          )

# model = LogisticRegression()

# model = BaggingClassifier(LogisticRegression(),
#                           n_estimators =100,
#                           n_jobs = -1,
#                           random_state=4444, 
#                           bootstrap=True,   # 디폴트, 중복 허용
#                         #   bootstrap=False     # 중복허용 안함.
#                           )


# model = BaggingClassifier(RandomForestClassifier(),
#                           n_estimators =100,
#                           n_jobs = -1,
#                           random_state=4444, 
#                         #   bootstrap=True,   # 디폴트, 중복 허용
#                           bootstrap=False     # 중복허용 안함.
#                           )

# model = RandomForestClassifier()

model = BaggingClassifier(RandomForestClassifier(),
                          n_estimators =100,
                          n_jobs = -1,
                          random_state=4444, 
                        #   bootstrap=True,   # 디폴트, 중복 허용
                          bootstrap=False     # 중복허용 안함.
                          )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score: ', acc)


# 디시전


# 디시전 배깅 부트스트랩 투루


# 디시전 배깅 부트스트랩 펄스


# 로지스틱


# 로지스틱 배깅, 부두스트랩 투루


# 로지스틱 배깅, 부투스트랩  펄스


# 랜포


# 랜포배깅, 부투스트랩  투루


# 랜포배깅, 부투스트랩  펄스