# 경우의 수를 100개로 늘려서 램덤서치!!!!
# 러닝레이트 반드시 넣고 다른 파라미터도 두어개 더 넣어라.

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import time

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=3333
)   # stratify=y

n_splits = 5

# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)

parameters = [
    {"n_jobs":[- 1], 'n_estimators':[100, 500], 'max_depth':[6, 10, 12],
     'min_samples_leaf' : [3, 10], 'learning_rate': [0.1, 0.01, 0.001],
     'subsample': [0.8, 1.0]}, # 36
    {"n_jobs":[-1,], 'max_depth':[6, 8, 10, 12],
     'min_samples_leaf' : [3, 5, 7, 10], 
     'learning_rate': [0.1, 0.01, 0.001], 'colsample_bytree': [0.8, 1.0]},   # 48
    {"n_jobs":[-1,], 'min_samples_leaf':[3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10],
     'learning_rate': [0.01, 0.001], 'subsample': [0.8, 1.0]}, # 64
    {'n_jobs' : [-1], 'min_samples_split' : [2, 3, 5, 10],
     'learning_rate': [0.01, 0.001], 'colsample_bytree': [0.8, 1.0]} ,  # 32   
]   # 총 180개의 조합

#2. 모델
model = RandomizedSearchCV(xgb.XGBRegressor(), 
                                    parameters, 
                                    cv=kfold, 
                                    verbose=1,
                                    refit=True,    # 제일 좋은 모델 한번더 돌림.
                                    n_jobs=-1, # CPU모든 코어 다 쓰라고 하는 것.
                                    n_iter=10, # GridSearch랑 다른 점1.실행하는 자 / 디폴트 10개
                                    random_state=3333, # GridSearch랑 다른 점.2
                                    )

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가.
print('최적의 매개변수 : ', model.best_estimator_)  
# 가장 좋은 성능을 내는 모델의 설정을 보여줍니다. 
# 이 모델은 우리가 찾은 최적의 파라미터로 구성되어 있습니다.

print('최적의 파라미터: ', model.best_params_)
# 모델의 성능을 가장 좋게 만드는 파라미터(설정값)를 보여줍니다.

print('best_score : ', model.best_score_)   
# 모델을 튜닝할 때 얻은 최고의 성적을 보여줍니다. 
# 이 성적은 보통 교차 검증 결과로 얻어진 평균 점수입니다.

print('model.score : ', model.score(x_test, y_test))   
# 테스트 데이터로 모델의 성능 점수를 계산해 보여줍니다. 
# 이 점수는 모델이 테스트 데이터에 대해 얼마나 잘 맞는지를 보여줍니다.

y_predict = model.predict(x_test)
# print('accuracy_score : ', accuracy_score(y_test, y_predict)) # 분류
# 모델이 예측한 결과(y_predict)와 실제 결과(y_test)를 비교해서 
# 정확도를 계산하고 보여줍니다.

print('Mean Squared Error : ', mean_squared_error(y_test, y_predict))
print('R^2 Score : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test) # 이게 제일 나음.
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))
# 최적의 파라미터로 설정된 모델의 정확도를 계산하고 보여줍니다. 
# 이는 최적화된 모델의 성능을 보여줍니다.

print('최적 튠 ACC : ', r2_score(y_test, y_pred_best))

print('최적의 learning_rate: ', model.best_params_['learning_rate'])

print('걸린시간: ', round(end_time - start_time,2), '초')


# 최적의 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.1, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=6, max_leaves=None,
#              min_child_weight=None, min_samples_leaf=10, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=100,
#              n_jobs=-1, num_parallel_tree=None, ...)
# 최적의 파라미터:  {'subsample': 0.8, 'n_jobs': -1, 'n_estimators': 100, 'min_samples_leaf': 10, 'max_depth': 6, 'learning_rate': 0.1}
# best_score :  0.8876686991826276
# model.score :  0.8873807419836552
# Mean Squared Error :  8.724819153160217
# R^2 Score :  0.8873807419836552
# 최적 튠 ACC :  0.8873807419836552
# 최적의 learning_rate:  0.1
# 걸린시간:  3.77 초