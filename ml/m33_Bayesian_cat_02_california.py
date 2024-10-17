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

from catboost import CatBoostRegressor

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
    model = CatBoostRegressor(
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

# {'target': 0.8467410498154921, 
# 'params': {'bagging_temperature': 1.1871123891752362, 
# 'border_count': 105.93348168692759, 
# 'depth': 12.0, 
# 'l2_leaf_reg': 4.624742044175747, 
# 'learning_rate': 0.2, 
# 'random_strength': 1.074040078988221}}
# 100 번 걸린시간 :  4527.0 초
