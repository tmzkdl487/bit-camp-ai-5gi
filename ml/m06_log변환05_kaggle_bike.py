# keras16_val5_kaggle_bike.py

# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv (카글 컴피티션 사이트)
# keras13_kaggle_bike1.py 수정

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#1. 데이터
path = 'C:/ai5/_data/kaggle/bike-sharing-demand/'  

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test2.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

x  = train_csv.drop(['count'], axis=1)   
y = train_csv['count']

# x.boxplot()
# plt.show()  # 'casual', 'registered'

y = pd.DataFrame(y)

# y.boxplot()
# plt.show()  

# exit()

######################## Population x 로그 변환 ###################
x[['casual', 'registered']] = np.log1p(x[['casual', 'registered']])   # 지수변환 np.exp1m / 로그, 지수 짝이 맞아야함
##############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=33)

######################## y 로그 변환 ###################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
#######################################################

#2. 모델 구성   모래시계 모형은 안됨.


#2. 모델 구성
model = LinearRegression()

#3. 컴파일, 훈련
model.fit(x_train, y_train, )

#4. 평가, 예측
score = model.score(x_test, y_test)   # r2_score와 같음

print('score : ', score)

y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)   # (10886,)

sampleSubmission['count'] = y_submit
# print(sampleSubmission)
# print(sampleSubmission.shape)   #(10886,)

sampleSubmission.to_csv(path + "sampleSubmission_0717_1523.csv")

### LR (LinearRegression)### score 1이 제일 좋음.
# 로그 변환 전 : score :  1.0
#  x 변환 후   : score :  0.6849247489328902
#  y 변환 후   : score :  0.6836701111206518
#  x, y 둘다   : score :  0.9960636428425416