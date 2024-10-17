import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression # <- 분류 모델, 
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

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
print('r2_score : ', r2)

# 디시전
# 최종점수 :  -0.1979034754639204
# r2_score : -0.1979034754639204

# 디시전 배깅 부트스트랩 투루
# 최종점수 :  0.48279172586722796
# r2_score :  0.48279172586722796

# 디시전 배깅 부트스트랩 펄스
# 최종점수 :  -0.18342750888969794
# r2_score :  -0.18342750888969794

# 리니어
# 최종점수 :  0.5262207027929591
# r2_score :  0.5262207027929591

# 리니어 배깅, 부두스트랩 투루
# 최종점수 :  0.524821185945685
# r2_score :  0.524821185945685

# 리니어 배깅, 부투스트랩  펄스
# 최종점수 :  0.5262207027929591
# r2_score :  0.5262207027929591

# 랜포
# 최종점수 :  0.4830098329496503
# r2_score :  0.4830098329496503

# 랜포배깅, 부투스트랩  투루
# 최종점수 :  0.5287592525821849
# r2_score :  0.5287592525821849

# 랜포배깅, 부투스트랩  펄스
# 최종점수 :  0.49717253613568513
# r2_score :  0.49717253613568513