import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

import xgboost as xgb
from xgboost import XGBClassifier, XGBRFRegressor

from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = 'C://ai5/_data/kaggle//bike-sharing-demand/'  

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)  # (10886, 11)
# print(test_csv.shape)   # (6493, 10)
# print(sampleSubmission.shape)   # (6493, 1)

########### x와 y를 분리
x  = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
# print(x)    # [10886 rows x 10 columns]

y = train_csv['count']

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    # stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params= {
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500), 
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth,
              num_leaves, min_child_samples, 
              min_child_weight, subsample, colsample_bytree,
              max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),                 # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),            # 0 ~ 1 사이의 값
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),           # 무조건 10 이상
        'reg_lambda' : max(reg_lambda, 0),                  # 무조건 양수만 
        'reg_alpha' : reg_alpha,
        
    }
    
    model = XGBRFRegressor(**params, n_jobs=-1)
    
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
            #   eval_metric='logloss',
              verbose=0,
              )
    y_predict = model.predict(x_test)
    # results = accuracy_score(y_test, y_predict)
    results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 : ', round(end_time - start_time, 2), '초')

# {'target': 0.05058556795120239, 
# 'params': {'colsample_bytree': 0.7508445798127961, 
# 'learning_rate': 0.09912378391612146, 
# 'max_bin': 233.4850308923398, 
# 'max_depth': 8.176489537660263, 
# 'min_child_samples': 81.06178248614937, 
# 'min_child_weight': 41.876258192018724, 
# 'num_leaves': 27.161005049498996, 
# 'reg_alpha': 37.91120340537671, 
# 'reg_lambda': 2.5891492898759796, 
# 'subsample': 0.6263000412624306}}
# 100 번 걸린시간 :  19.01 초