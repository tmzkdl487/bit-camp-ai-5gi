import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = 'C://ai5/_data/kaggle//bike-sharing-demand/'  

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

x  = train_csv.drop(['casual', 'registered', 'count'], axis=1)   

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    shuffle=True, 
                                                    random_state=3333,
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

# {'target': 0.3984776735305786, 
# 'params': {'colsample_bytree': 1.0, 
# 'learning_rate': 0.1, 
# 'max_bin': 466.9720976730107, 
# 'max_depth': 10.0, 
# 'min_child_samples': 92.67003044517622, 
# 'min_child_weight': 1.0, 
# 'num_leaves': 30.86524472728791, 
# 'reg_alpha': 44.15870624561117, 
# 'reg_lambda': 10.0, 
# 'subsample': 0.9614105580898901}}
# 500 번 걸린시간 :  542.98 초




