# keras23_kaggle1_santander_customer.py

# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

# 맹그러!!!
# 다중분류인줄 알았더니 이진분류였다!!!
# 다중분류 다시 찾겠노라!!!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#1. 데이터
path = 'C://ai5/_data/kaggle/santander-customer-transaction-prediction/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())

x  = train_csv.drop(['target'], axis=1) 
y = train_csv['target']

# print(x.columns)
# Index(['var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7',
#        'var_8', 'var_9',
#        ...
#        'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
#        'var_196', 'var_197', 'var_198', 'var_199'],
#       dtype='object', length=200)

# x.boxplot()
# plt.show()  # 'var_45', 'var_61', 'var_90', 'var_187'

y = pd.DataFrame(y)

# y.boxplot()
# plt.show()  

# exit()

# 'var_45'

####################### Population x 로그 변환 ###################
x[['var_45', 'var_61', 'var_90', 'var_187']] = np.log1p(x[['var_45', 'var_61', 'var_90', 'var_187']])   # 에러
#############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True,
                                                    random_state=3,
                                                    stratify=y)

######################## y 로그 변환 ###################
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
#######################################################

#2. 모델
model = LinearRegression()

#3. 컴파일, 훈련
model.fit(x_train, y_train, )

#4. 평가, 예측
score = model.score(x_test, y_test)   # r2_score와 같음

print('score : ', score)

y_submit = model.predict(test_csv)

sample_submission_csv['target'] = y_submit

sample_submission_csv.to_csv(path + "sampleSubmission_0724_1630.csv")

# print("로스는 : ", round(loss[0], 4))
# print("ACC : ", round(loss[1], 3))
# print("걸린시간: " , round(end_time - start_time, 2), "초")

# ACC :  0.383
# ACC :  0.9

### LR (LinearRegression)###
# 로그 변환 전 : score :  0.18207788643809508
#  x 변환 후   : ValueError: Input contains NaN, infinity or a value too large for dtype('float64'). / 음수라서 안되나봄. -> 선생님이 스켈링하라고 하심.
#  y 변환 후   : score :  0.18207788643809475
#  x, y 둘다   : ValueError: Input contains NaN, infinity or a value too large for dtype('float64').