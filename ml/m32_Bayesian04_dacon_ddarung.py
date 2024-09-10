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

from xgboost import XGBClassifier, XGBRegressor
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

# submission_csv = pd.read_csv(path + "submission.csv", index_col=0)  

train_csv = train_csv.dropna()  # 구멍난 데이터를 삭제해달라는 수식

test_csv = test_csv.fillna(test_csv.mean()) 

x = train_csv.drop(['count'], axis=1)   # train_csv에서 count 지우는 수식을 만들고 있다. count 컬럼의 axis는 가로 1줄을 지운다. 행을 지운다. []안해도 나온다.
# print(x)    # [1328 rows x 9 columns] / 확인해봄.
# y = train_csv['count'].astype(int)  # y는 count 열만 가지고 옴. y를 만들고 있다.
# print(y.shape)  # (1328,)   # 확인해봄.

# print(x.shape, y.shape) # (1328, 9) (1328,)

y = train_csv['count']

# exit()

# y 값이 연속적인 값이 아니어서 에러가 발생하므로 이를 해결하기 위해 다음과 같이 처리
# y_unique = np.unique(y)
# y_map = {val: idx for idx, val in enumerate(y_unique)}
# y = y.map(y_map)  # y 값의 고유값을 0부터 연속적으로 변경

# pca = PCA(n_components=8)  
# x = pca.fit_transform(x)

# evr = pca.explained_variance_ratio_ 

# cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 1
# print(np.argmax(cumsum >= 0.99) +1)  # 1
# print(np.argmax(cumsum >= 0.999) +1) # 3
# print(np.argmax(cumsum >= 1.0) +1)   # 1

# print(np.unique(y))
# exit()

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, 
                                                    random_state=random_state,
                                                    shuffle=True,
                                                    # stratify=y,
                                                    ) 

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)

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
        'max_depth': int(round(max_depth)),                 # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),            # 0 ~ 1 사이의 값
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),           # 무조건 10 이상
        'reg_lambda' : max(reg_lambda, 0),                  # 무조건 양수만 
        'reg_alpha' : reg_alpha,
        
    }
    
    model = XGBRegressor(**params, n_jobs=-1)
    
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

# {'target': 0.7237052048037198, 
# 'params': {'colsample_bytree': 0.841477324882826, 
# 'learning_rate': 0.05319380440854934, 
# 'max_bin': 75.41743623225658, 
# 'max_depth': 9.17252124215784, 
# 'min_child_samples': 16.92852974036208, 
# 'min_child_weight': 8.900313917092333, 
# 'num_leaves': 33.56963181520658, 
# 'reg_alpha': 47.57276733924565, 
# 'reg_lambda': 6.405530178914074, 
# 'subsample': 0.9704042829769346}}
# 100 번 걸린시간 :  26.02 초