import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import time

import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = 'C://ai5//_data//kaggle//otto-group-product-classification-challenge//'

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 분류
# print(train_csv)    # [61878 rows x 94 columns]
 
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
# print(test_csv)   # [144368 rows x 93 columns]
    
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(train_csv.shape, test_csv.shape, sampleSubmission_csv.shape)
# (61878, 94) (144368, 93) (144368, 9)

# [누리님 조언] 타겟을 숫자로 바꾼다.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
# print(x)    # [61878 rows x 93 columns]

y = train_csv['target']
# print(y.shape)  # (61878,)

y_ohe = pd.get_dummies(y)
# print(y_ohe.shape) 

# print(x.shape, y.shape) # (61878, 93) (61878,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=3333, stratify=y
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
    
    model = XGBClassifier(**params, n_jobs=-1,
                            tree_method='hist',
                            device='cuda',
                            # eval_metric='mlogloss',
                            callbacks=[early_stop],)
    
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
            #   eval_metric='mlogloss',
              verbose=0,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
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

# {'target': 0.8235294117647058, 
# 'params': {'colsample_bytree': 0.9433726102176561,
# 'learning_rate': 0.1, 
# 'max_bin': 64.25065410099377, 
# 'max_depth': 10.0, 
# 'min_child_samples': 24.03562733326326, 
# 'min_child_weight': 1.0, 
# 'num_leaves': 40.0, 
# 'reg_alpha': 0.01, 
# 'reg_lambda': -0.001, 
# 'subsample': 0.5}}
# 100 번 걸린시간 :  900.76 초