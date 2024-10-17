# keras23_kaggle2_otto.py

# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#1. 데이터
path = 'C://ai5//_data//kaggle//otto-group-product-classification-challenge//'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
 
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
    
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])

x = train_csv.drop(['target'], axis=1)

y = train_csv['target']

y_ohe = pd.get_dummies(y)

# x.boxplot()
# plt.show()  # 'feat_73', 'feat_74', 'feat_90'

y = pd.DataFrame(y)

# y.boxplot()
# plt.show() 

####################### Population x 로그 변환 ###################
x[['feat_73', 'feat_74', 'feat_90']] = np.log1p(x[['feat_73', 'feat_74', 'feat_90']])   # 에러
#############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.6, shuffle=True, 
                                                    random_state=3, 
                                                    stratify=y)

# print(x_train.shape, y_train.shape) # (46408, 93) (46408,)
# print(x_test.shape, y_test.shape)   # (15470, 93) (15470,)

######################## y 로그 변환 ###################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
#######################################################

#2. 모델 구성
model = LinearRegression()

#3. 컴파일, 훈련
model.fit(x_train, y_train, )

#4. 평가, 예측
score = model.score(x_test, y_test)   # r2_score와 같음

print('score : ', score)

y_submit = model.predict(test_csv)

y_submit = np.round(y_submit)

sampleSubmission_csv[['Class_1','Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit

sampleSubmission_csv.to_csv(path + "sampleSubmission_0724_2130.csv")

# print("로스는 : ", round(loss[0], 4))
# print("ACC : ", round(loss[1], 3))
# print("걸린시간: " , round(end_time - start_time, 2), "초")
    
# [실습] ACC 0.89 이상 메일로 선생님께 보내드리기 -> 점수로 바꿈.
# ACC :  0.769

### LR (LinearRegression)###
# 로그 변환 전 : score :  0.3845255168419896
#  x 변환 후   : score :  0.38791959006636056
#  y 변환 후   : score :  0.384542254458214
#  x, y 둘다   : score :  0.3879350575360521