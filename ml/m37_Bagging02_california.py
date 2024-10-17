import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression # <- 분류 모델, 
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    random_state=4444,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    # stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeRegressor()

# model = BaggingRegressor(DecisionTreeRegressor(),
#                           n_estimators =100,
#                           n_jobs = -1,
#                           random_state=4444, 
#                         #   bootstrap=True,   # 디폴트, 중복 허용
#                           bootstrap=False     # 중복허용 안함.
#                           )

# model = LinearRegression()

# model = BaggingRegressor(LinearRegression(),
#                           n_estimators =100,
#                           n_jobs = -1,
#                           random_state=4444, 
#                         #   bootstrap=True,   # 디폴트, 중복 허용
#                           bootstrap=False     # 중복허용 안함.
#                           )

# model = RandomForestRegressor()

model = BaggingRegressor(RandomForestRegressor(),
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
# acc = accuracy_score(y_test, y_predict)
# print('acc_score: ', acc)

r2 = r2_score(y_test, y_predict)
print('r2_score', r2)

# 디시전
# 최종점수 :  0.610631086996443
# r2_score 0.610631086996443

# 디시전 배깅 부트스트랩 투루
# 최종점수 :  0.8107197085695325
# r2_score 0.8107197085695325

# 디시전 배깅 부트스트랩 펄스
# 최종점수 :  0.6302361114247275
# r2_score 0.6302361114247275

# 로지스틱
# 최종점수 :  0.6011614863584488
# r2_score 0.6011614863584488

# 로지스틱 배깅, 부두스트랩 투루
# 최종점수 :  0.5962043412746386
# r2_score 0.5962043412746386

# 로지스틱 배깅, 부투스트랩  펄스
# 최종점수 :  0.6011614863584488
# r2_score 0.6011614863584488

# 랜포
# 최종점수 :  0.8104507999055205
# r2_score 0.8104507999055205

# 랜포배깅, 부투스트랩  투루
# 최종점수 :  0.8075877878538014
# r2_score 0.8075877878538014

# 랜포배깅, 부투스트랩  펄스
# 최종점수 :  0.8145608091094456
# r2_score 0.8145608091094456