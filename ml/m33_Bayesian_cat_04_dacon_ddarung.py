from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
import sklearn as sk
import time

from xgboost import XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "C://ai5//_data//dacon//따릉이//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 인덱스 없으면 index_col쓰면 안됨. 0은 0번째 줄 없앴다는 뜻이다.
# print(train_csv)    # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)  

train_csv = train_csv.dropna()  # 구멍난 데이터를 삭제해달라는 수식

test_csv = test_csv.fillna(test_csv.mean()) 

x = train_csv.drop(['count'], axis=1)   # train_csv에서 count 지우는 수식을 만들고 있다. count 컬럼의 axis는 가로 1줄을 지운다. 행을 지운다. []안해도 나온다.
# print(x)    # [1328 rows x 9 columns] / 확인해봄.
y = train_csv['count']  # y는 count 열만 가지고 옴. y를 만들고 있다.
# print(y.shape)  # (1328,)   # 확인해봄.

# print(x.shape, y.shape) # (1328, 9) (1328,)

# exit()

pca = PCA(n_components=8)  
x = pca.fit_transform(x)

# evr = pca.explained_variance_ratio_ 

# cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 1
# print(np.argmax(cumsum >= 0.99) +1)  # 1
# print(np.argmax(cumsum >= 0.999) +1) # 3
# print(np.argmax(cumsum >= 1.0) +1)   # 1

# exit()

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    random_state=random_state,
                                                    shuffle=True,
                                                    # stratify=y,
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

# {'target': 0.746498305736583, 
# 'params': {'bagging_temperature': 3.165142184084975, 
# 'border_count': 77.01223733872965, 
# 'depth': 6.892718132964249, 
# 'l2_leaf_reg': 7.035818718071639, 
# 'learning_rate': 0.19824575479023515, 
# 'random_strength': 9.603381410678251}}
# 100 번 걸린시간 :  2858.48 초

