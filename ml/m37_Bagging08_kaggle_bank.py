import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression # <- 분류 모델
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
# print(x)                            # [165034 rows x 10 columns]
y = train_csv['Exited']
# print(y.shape)                      # (165034,)

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()

# model = BaggingClassifier(DecisionTreeClassifier(),
#                           n_estimators =100,
#                           n_jobs = -1,
#                           random_state=4444, 
#                           bootstrap=True,   # 디폴트, 중복 허용
#                           # bootstrap=False     # 중복허용 안함.
#                           )

# model = LogisticRegression()

# model = BaggingClassifier(LogisticRegression(),
#                           n_estimators =100,
#                           n_jobs = -1,
#                           random_state=4444, 
#                           # bootstrap=True,   # 디폴트, 중복 허용
#                           bootstrap=False     # 중복허용 안함.
#                           )

# model = RandomForestClassifier()

model = BaggingClassifier(RandomForestClassifier(),
                          n_estimators =100,
                          n_jobs = -1,
                          random_state=4444, 
                          # bootstrap=True,   # 디폴트, 중복 허용
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
# 최종점수 :  0.7957100009088981
# acc_score:  0.7957100009088981

# 디시전 배깅 부트스트랩 투루
# 최종점수 :  0.8553943102978157
# acc_score:  0.8553943102978157

# 디시전 배깅 부트스트랩 펄스
# 최종점수 :  0.7984366952464629
# acc_score:  0.7984366952464629

# 로지스틱
# 최종점수 :  0.8243705880570789
# acc_score:  0.8243705880570789

# 로지스틱 배깅, 부두스트랩 투루
# 최종점수 :  0.8244311812645803
# acc_score:  0.8244311812645803

# 로지스틱 배깅, 부투스트랩  펄스
# 최종점수 :  0.8243705880570789
# acc_score:  0.8243705880570789

# 랜포
# 최종점수 :  0.8585754536916411
# acc_score:  0.8585754536916411

# 랜포배깅, 부투스트랩  투루
# 최종점수 :  0.861726300481716
# acc_score:  0.861726300481716

# 랜포배깅, 부투스트랩  펄스
# OSError: [WinError 1450] 시스템 리소스가 부족하기 때문에 요청한 서비스를 완성할 수 없습니다