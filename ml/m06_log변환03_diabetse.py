# keras16_val3_diabetes.py 복사

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target

# print(df)   # [442 rows x 11 columns]

# df.boxplot()
# plt.show()    # target만 엄청 이상함.

# exit()

x = datasets.data
y = datasets.target 

# print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=8989)

######################## y 로그 변환 ###################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
#######################################################

#2. 모델 구성
model = RandomForestRegressor(random_state=1234,
                              max_depth=5,   # 5가 디폴트
                              min_samples_split=3)   # 모두 동일한 파라미터 사용함

#3. 컴파일, 훈련
model.fit(x_train, y_train, )

#4. 평가, 예측
score = model.score(x_test, y_test)   # r2_score와 같음

print('score : ', score)


y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)

# [실습] 맹그러봐 / R2 0.62 이상
# r2스코어 :  0.5189042303718636
# r2스코어 :  0.5665473691683233
# r2스코어 :  0.6007596386970162
# r2스코어 :  0.6006874672342544
# r2스코어 :  0.6151010366297509
# r2스코어 :  0.6180262183555738
# r2스코어 :  0.6196877963446061
# r2스코어 :  0.6211853813430838

############### y 로그변환
# score :  0.386028582827629 / r2 :  0.386028582827629