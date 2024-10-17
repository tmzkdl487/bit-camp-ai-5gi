# keras16_val4_dacon_ddarung.py 복사

# https://dacon.io/competitions/open/235576/overview/description (대회 사이트 주소)

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#1. 데이터
path = "C:/ai5/_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 인덱스 없으면 index_col쓰면 안됨. 0은 0번째 줄 없앴다는 뜻이다.
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)

train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1) 
y = train_csv['count']

# x.boxplot()
# plt.show()  # hour_bef_visibility

# exit()

y = pd.DataFrame(y)

# y.boxplot()
# plt.show()  

######################## Population x 로그 변환 ###################
x['hour_bef_visibility'] = np.log1p(x['hour_bef_visibility'])   # 지수변환 np.exp1m / 로그, 지수 짝이 맞아야함
##############################################################


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.1, random_state=512) # random_state=3454, 맛집 레시피 : 4343

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

y_submit = model.predict(test_csv)  # 예측한 값을 y_submit에 넣는다는 뜻.
# print(y_submit) # 확인 해봄.
# print(y_submit.shape)   # (715, 1) / 나왔음.

###################### submissinon.csv만들기 // count컬럼에 값만 넣으주면 돼. ##########

submission_csv['count'] = y_submit  # submission count 열에 y_submit을 넣겠다는 수식.
# print(submission_csv)   # 확인
# print(submission_csv.shape) #.shape확인.

submission_csv.to_csv(path + "submission_0717_0944.csv")    # 폴더 안에 파일로 만들겠다. 가로 안은 (저장 경로 + 파일명)이다.

# print ("로스는 : ", loss)   # 확인하려고 마지막에 집어넣음. 

# 로스는 :  2859.677734375 / # 로스는 :  2744.799072265625 / # 로스는 :  2741.8271484375 / # 로스는 :  2724.955810546875
# 로스는 :  2949.488525390625 / # 로스는 :  2766.807861328125 / # 로스는 :  2740.542724609375 / # 로스는 :  2640.291748046875
# 로스는 :  2604.446044921875 / # 로스는 :  2608.508544921875 / 로스는 :  2588.36962890625 / 로스는 :  2584.096923828125
# 로스는 :  2299.7529296875 / 로스는 :  2264.956787109375 / 로스는 :  1717.86669921875 / 로스는 :  1731.430419921875
# 로스는: 1696.13525390625 / 


### LR (LinearRegression)###
# 로그 변환 전 : score :  0.5380704230891991
#  x 변환 후   : score :  0.5371087550534873
#  y 변환 후   : score :  0.5114883438641866
#  x, y 둘다   : score :  0.5091505403318493

### LR (LinearRegression)###
# 로그 변환 전 : 
#  x 변환 후   :
#  y 변환 후   :
#  x, y 둘다   : 