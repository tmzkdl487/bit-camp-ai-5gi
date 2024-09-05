import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, 
                                                    random_state= 3377, 
                                                    stratify=y,
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import xgboost as xgb
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name='mlogloss', # 이진: logloss, 다중: mlogloss, error
    data_name='validation_0',
    # save_best=True,   # ??????? 이거 넣으면 AttributeError: `best_iteration` is only defined when early stopping is used. 에러남.
)

#2. 모델
model = XGBClassifier(
    n_estimators = 1000,
    max_depth = 8,
    gamma = 0,
    min_child_weight= 0,
    subsample= 0.4,
    reg_alpha= 0,    # L1 규제 리소
    reg_lambda= 1,   # L2 규제 라지
    # eval_metric='mlogloss', # 2.1.1 버전에서 컴파일이 아니라 모델로 가야됨.. / 이진분류에서error도 쓸 수 있음.
    callbacks=[early_stop], 
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
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_)    # 오름차순.
print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    select_model = XGBClassifier(  n_estimators = 1000,
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
    score = accuracy_score(y_test, select_y_predict)
    
    print('Trech=%.3f, n=%d, ACC:%.2f%%' %(i, select_x_train.shape[1], score*100))

# 최종점수 :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# [0.09076571 0.05376026 0.018864   0.03917528 0.14172934 0.03673201
#  0.17493507 0.00297603 0.00193168 0.15767676 0.12651627 0.01215809
#  0.14277951]
# [0.00193168 0.00297603 0.01215809 0.018864   0.03673201 0.03917528
#  0.05376026 0.09076571 0.12651627 0.14172934 0.14277951 0.15767676
#  0.17493507]
# Trech=0.002, n=13, ACC:97.22%
# Trech=0.003, n=12, ACC:97.22%
# Trech=0.012, n=11, ACC:97.22%
# Trech=0.019, n=10, ACC:97.22%
# Trech=0.037, n=9, ACC:94.44%
# Trech=0.039, n=8, ACC:94.44%
# Trech=0.054, n=7, ACC:94.44%
# Trech=0.091, n=6, ACC:97.22%
# Trech=0.127, n=5, ACC:97.22%
# Trech=0.142, n=4, ACC:97.22%
# Trech=0.143, n=3, ACC:97.22%
# Trech=0.158, n=2, ACC:97.22%
# Trech=0.175, n=1, ACC:66.67%