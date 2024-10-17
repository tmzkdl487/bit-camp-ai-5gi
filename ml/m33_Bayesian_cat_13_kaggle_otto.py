import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import time

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier
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
bayesian_params = {
    'learning_rate': (0.01, 0.2),  
    'depth': (4, 12),             
    'l2_leaf_reg': (1, 10),       
    'bagging_temperature': (0.0, 5.0),  
    'border_count': (32, 255),  
    'random_strength': (1, 10),   
}
def cat_hamsu(learning_rate, depth,
              l2_leaf_reg, bagging_temperature, 
              border_count, random_strength):
    
    params = {
        'learning_rate' : learning_rate,
        'depth' : int(round(depth)),                 
        'l2_leaf_reg' : int(round(l2_leaf_reg)),
        'bagging_temperature' : bagging_temperature,
        'border_count' : int(round(border_count)),
        'random_strength' : random_strength,           
    }
    
    # cat_features = list(range(x_train.shape[1]))
    
    # 2. 모델
    model = CatBoostClassifier(
        **params,
        iterations=500,            # 트리 개수 (기본값: 500)
        task_type="GPU",            # GPU 사용 (기본값: 'CPU')
        devices='0',                # 첫번째 GPU 사용 (기본값: 모든 GPU 가용)
        early_stopping_rounds=100,  # 조기 종료 (기본값:  None)
        verbose=10,                 # 매 10번째 반복마다 출력 (기본값: 100)
        # cat_features = cat_features
    )
    
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              verbose=0,
              )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    # results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f=cat_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 : ', round(end_time - start_time, 2), '초')

# {'target': 0.8236102133160956, 
# 'params': {'bagging_temperature': 0.0, 
# 'border_count': 182.38904814565157, 
# 'depth': 12.0, 
# 'l2_leaf_reg': 9.935682214426203, 
# 'learning_rate': 0.2, 
# 'random_strength': 1.0}}
# 100 번 걸린시간 :  5545.5 초
