import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)
# print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, 
                                                    random_state= 3377, 
                                                    # stratify=y,
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import xgboost as xgb
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name='logloss', # 이진: logloss, 다중: mlogloss, error
    data_name='validation_0',
    # save_best=True,   # ??????? 이거 넣으면 AttributeError: `best_iteration` is only defined when early stopping is used. 에러남.
)

#2. 모델
model = XGBRFRegressor(
    n_estimators = 1000,
    max_depth = 8,
    gamma = 0,
    min_child_weight= 0,
    subsample= 0.4,
    reg_alpha= 0,    # L1 규제 리소
    reg_lambda= 1,   # L2 규제 라지
    # eval_metric='mlogloss', # 2.1.1 버전에서 컴파일이 아니라 모델로 가야됨.. / 이진분류에서error도 쓸 수 있음.
    # callbacks=[early_stop], 
    random_state=3377,
)

#3. 컴파일, 훈련
model.fit(x_train, y_train,
         eval_set=[(x_test, y_test)],
         verbose=1,
        #   eval_metric='mlogloss', # 2.1.1 버전에서 위로감.
          )

#4. 평가, 예측
results =  model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print('accuracy_score : ', acc)

r2 = r2_score(y_test, y_predict)
print('accuracy_score : ', r2)

print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_)    # 오름차순.
print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    select_model = XGBRFRegressor(  n_estimators = 1000,
                                    max_depth = 8,
                                    gamma = 0,
                                    min_child_weight= 0,
                                    subsample= 0.4,
                                    # reg_alpha= 0,    # L1 규제 리소
                                    # reg_lambda= 1,   # L2 규제 라지
                                    # eval_metric='mlogloss', # 2.1.1 버전에서 컴파일이 아니라 모델로 가야됨.. / 이진분류에서error도 쓸 수 있음.
                                    # callbacks=[early_stop], 
                                    random_state=3377,)
    
    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_test, y_test)],
                     verbose=0,
                     )
    select_y_predict = select_model.predict(select_x_test)
    # score = accuracy_score(y_test, select_y_predict)
    r2 = r2_score(y_test, select_y_predict)
    
    print('Trech=%.3f, n=%d, ACC:%.2f%%' %(i, select_x_train.shape[1],r2*100)) 
    
# [0]     validation_0-rmse:61.23789
# 최종점수 :  0.3634173639891851
# accuracy_score :  0.3634173639891851
# [0.02414967 0.02892634 0.21431467 0.07682849 0.04716727 0.05588329
#  0.07344671 0.09679318 0.31569782 0.06679252]
# [0.02414967 0.02892634 0.04716727 0.05588329 0.06679252 0.07344671
#  0.07682849 0.09679318 0.21431467 0.31569782]
# Trech=0.024, n=10, ACC:36.68%
# Trech=0.029, n=9, ACC:35.09%
# Trech=0.047, n=8, ACC:33.95%
# Trech=0.056, n=7, ACC:35.07%
# Trech=0.067, n=6, ACC:33.93%
# Trech=0.073, n=5, ACC:32.62%
# Trech=0.077, n=4, ACC:36.18%
# Trech=0.097, n=3, ACC:33.08%
# Trech=0.214, n=2, ACC:33.70%
# Trech=0.316, n=1, ACC:20.80%