# https://dacon.io/competitions/open/235576/overview/description (대회 사이트 주소)

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "C:/ai5/_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 인덱스 없으면 index_col쓰면 안됨. 0은 0번째 줄 없앴다는 뜻이다.
# print(train_csv)    # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
# print(submission_csv)   # [715 rows x 1 columns]

# print(train_csv.shape)  # (1459, 10)
# print(test_csv.shape) # (715, 9)
# print(submission_csv.shape) # (715, 1)

# print(train_csv.columns)    # 열의 이름을 알려달라는 수식. 
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info()) # Non-Null이 몇갠지 구멍난 데이터가 있는지 확인하는 수식.

################# 결측치 처리 1. 삭제 ################### 행은 다 같아야 함.
# print(train_csv.isnull().sum()) 밑에 코드도 같은 코드다. isnull이나 isna나 똑같다.
print(train_csv.isna().sum())   # 구멍난 데이터의 수를 알려달라.   

train_csv = train_csv.dropna()  # 구멍난 데이터를 삭제해달라는 수식
print(train_csv.isna().sum())   # 잘 지워졌는지 확인.

print(train_csv)    # [1328 rows x 10 columns]
print(train_csv.isna().sum())   # 다시 확인
print(train_csv.info()) # 다시 확인

print(test_csv.info())  # 이제 test 파일도 정보확인

test_csv = test_csv.fillna(test_csv.mean()) # 구멍난 데이터를 평균값으로 채워달라는 뜻.
print(test_csv.info()) # 715 non-nul /  확인.

x = train_csv.drop(['count'], axis=1)   # train_csv에서 count 지우는 수식을 만들고 있다. count 컬럼의 axis는 가로 1줄을 지운다. 행을 지운다. []안해도 나온다.
print(x)    # [1328 rows x 9 columns] / 확인해봄.
y = train_csv['count']  # y는 count 열만 가지고 옴. y를 만들고 있다.
print(y.shape)  # (1328,)   # 확인해봄.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.1, random_state=512) # random_state=3454, 맛집 레시피 : 4343

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=8)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

y_submit = model.predict(test_csv)  # 예측한 값을 y_submit에 넣는다는 뜻.
print(y_submit) # 확인 해봄.
print(y_submit.shape)   # (715, 1) / 나왔음.

###################### submissinon.csv만들기 // count컬럼에 값만 넣으주면 돼. ##########

submission_csv['count'] = y_submit  # submission count 열에 y_submit을 넣겠다는 수식.
print(submission_csv)   # 확인
print(submission_csv.shape) #.shape확인.

submission_csv.to_csv(path + "submission_0717_0944.csv")    # 폴더 안에 파일로 만들겠다. 가로 안은 (저장 경로 + 파일명)이다.

print ("로스는 : ", loss)   # 확인하려고 마지막에 집어넣음. 

# 로스는 :  2859.677734375 / # 로스는 :  2744.799072265625 / # 로스는 :  2741.8271484375 / # 로스는 :  2724.955810546875
# 로스는 :  2949.488525390625 / # 로스는 :  2766.807861328125 / # 로스는 :  2740.542724609375 / # 로스는 :  2640.291748046875
# 로스는 :  2604.446044921875 / # 로스는 :  2608.508544921875 / 로스는 :  2588.36962890625 / 로스는 :  2584.096923828125
# 로스는 :  2299.7529296875 / 로스는 :  2264.956787109375 / 로스는 :  1717.86669921875 / 로스는 :  1731.430419921875
# 로스는: 1696.13525390625 / 