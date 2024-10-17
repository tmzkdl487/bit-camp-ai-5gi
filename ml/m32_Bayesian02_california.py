import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

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
    
    import xgboost as xgb
    early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    # metric_name='mlogloss', # 또는 error
    data_name='validation_0',
    save_best=True,
    )
    
    model = XGBRegressor(**params, n_jobs=-1,
                            tree_method='hist',
                            device='cuda',
                            # eval_metric='mlogloss',
                            callbacks=[early_stop],)
    
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

n_iter = 500
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 : ', round(end_time - start_time, 2), '초')

# {'target': 0.8436190845113714, 
# 'params': 
# {'colsample_bytree': 0.8200455412009147, 
# 'learning_rate': 0.1,
# 'max_bin': 132.75130327378758, 
# 'max_depth': 10.0, 
# 'min_child_samples': 181.10264016071278, 
# 'min_child_weight': 40.11596063756949, 
# 'num_leaves': 28.918322162380235, 
# 'reg_alpha': 0.01, 
# 'reg_lambda': 9.935375084796, 
# 'subsample': 0.8233278955102336}}
# 500 번 걸린시간 :  571.6 초

