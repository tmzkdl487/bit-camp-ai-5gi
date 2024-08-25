from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
# print(df)   # [20640 rows x 8 columns]

# df.boxplot()    # df.plot.box()랑 똑같음. 
# df.plot.box()   # 이건 선이 없이 나옴.
# plt.show()

# print(df.info())  # 범위 인덱스
#  0   MedInc      20640 non-null  float64
#  1   HouseAge    20640 non-null  float64
#  2   AveRooms    20640 non-null  float64
#  3   AveBedrms   20640 non-null  float64
#  4   Population  20640 non-null  float64
#  5   AveOccup    20640 non-null  float64
#  6   Latitude    20640 non-null  float64
#  7   Longitude   20640 non-null  float64
#  8   target      20640 non-null  float64

# print(df.describe())
#              MedInc      HouseAge      AveRooms     AveBedrms    Population      AveOccup      Latitude     Longitude        target
# count  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000
# mean       3.870671     28.639486      5.429000      1.096675   1425.476744      3.070655     35.631861   -119.569704      2.068558
# std        1.899822     12.585558      2.474173      0.473911   1132.462122     10.386050      2.135952      2.003532      1.153956
# min        0.499900      1.000000      0.846154      0.333333      3.000000      0.692308     32.540000   -124.350000      0.149990
# 25%        2.563400     18.000000      4.440716      1.006079    787.000000      2.429741     33.930000   -121.800000      1.196000
# 50%        3.534800     29.000000      5.229129      1.048780   1166.000000      2.818116     34.260000   -118.490000      1.797000
# 75%        4.743250     37.000000      6.052381      1.099526   1725.000000      3.282261     37.710000   -118.010000      2.647250
# max       15.000100     52.000000    141.909091     34.066667  35682.000000   1243.333333     41.950000   -114.310000      5.000010

# df['Population'].boxplot()   # 'Series' object has no attribute 'boxplot' 데이터프라임은 시리지
# df['Population'].plot.box()   # 이거 되
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()
# 스탠다드 스케일러 안하는 이유는 너무 커서
# 너무 클 때 생각해 볼건 y. ysms 직방이기 때문. x도 됨

# df['target'].hist(bins=50)
# plt.show()


# x와 y 분류
x = df.drop(['target'], axis=1).copy()
y = df['target']

######################## Population x 로그 변환 ###################
x['Population'] = np.log1p(x['Population'])   # 지수변환 np.exp1m / 로그, 지수 짝이 맞아야함
##############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234)


######################## y 로그 변환 ###################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
#######################################################


#2. 모델
# model = RandomForestRegressor(random_state=1234,
#                               max_depth=5,   # 5가 디폴트
#                               min_samples_split=3)   # 모두 동일한 파라미터 사용함

model = LinearRegression()

#3. 훈련
model.fit(x_train, y_train, )


#4. 평가 예측
score = model.score(x_test, y_test)   # r2_score와 같음

print('score : ', score)


y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)


# 로그 변환전 : score : 
# score :  0.6495152533878351
# r2 :  0.6495152533878351

# x만 로그변환
# score :  0.6495031475648194 / r2 :  0.6495031475648194

# x, y 로그변환
# score :  0.6584197269397019 / r2 :  0.6584197269397019

# LR (LinearRegression)
# 로그 변환 전 : 0.606572212210644
# 로그 변환 후 : 0.6295290651919587
# x에 pop 변환 후: score :  0.6495031475648194
# x, y 둘다:  score :  0.6584197269397019 

