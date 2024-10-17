import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

from catboost import CatBoostRegressor, CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
# print(x)                            # [165034 rows x 10 columns]
y = train_csv['Exited']
# print(y.shape)                      # (165034,)

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    stratify=y)

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
    # results = accuracy_score(y_test, y_predict)
    results = r2_score(y_test, y_predict)
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

# {'target': 0.1963670486840018, 
# 'params': {'bagging_temperature': 1.193700568007332,
# 'border_count': 77.42492029447048, 
# 'depth': 5.3962689657425225, 
# 'l2_leaf_reg': 4.738150871401265, 
# 'learning_rate': 0.1889302424497913, 
# 'random_strength': 5.202237528617707}}
# 100 번 걸린시간 :  460.04 초