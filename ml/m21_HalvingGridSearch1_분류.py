import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV 
# HalvingGridSearchCV 쓰려면은 enable_halving_search_cv을 꼭 위에 임포트해야된다.
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=15426, shuffle=True, train_size=0.8, stratify=y)

# print(x_train.shape, y_train.shape) # (1437, 64) (1437,)
# print(x_test.shape, y_test.shape)   # (360, 64) (360,)

n_splits = 5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    {'learning_rate' : [0.01, 0.05, 0.1,],# 0.2, 0.3], 
     'max_depth' : [3 , 4 ,5,],}, # 6, 8],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma' : [0, 0.1, 0.2, 0.5, 1.0],},
   ] # 5*5*4

#2. 모델
model = HalvingGridSearchCV(XGBClassifier(
                                          # tree_method='gpu_hist'
                                          tree_method='hist',
                                          device='cuda',
                                          n_estimators=50,
    ),
                     parameters, 
                     cv=kfold,
                     verbose=1, # 1이 아니고 2나 3이 좋아.
                     refit=True,
                    #  n_jobs=-1,
                    #  n_iter=10,
                     random_state=4325,
                     factor= 3.2,   #  HalvingGridSearchCV는 성능이 가장 낮은 절반의 조합을 제거합니다. factor=2이므로 절반으로 줄어듭니다.
                     min_resources= 30,
                     max_resources= 1437,
                     aggressive_elimination=True,
                     )

#3. 훈련
start_time = time.time()
model.fit(x_train,y_train,
          verbose=False,
          eval_set=[(x_test, y_test)],)
end_time = time.time()

#4. 예측
print('최적의 매개변수 :', model.best_estimator_)

print('최적의 파라미터',model.best_params_)

print('best_score : ', model.best_score_)

print('model.score :', model.score(x_test,y_test))

y_pred = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test,y_pred))  # 이전과 차이를 보기위해

y_pred_best = model.best_estimator_.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_pred_best))

print('걸린 시간 :', round(end_time - start_time, 2), '초')

### enable_halving_search_cv ### factor=2
# TypeError: 'module' object is not callable

### HalvingGridSearchCV ### 
# n_iterations: 4               / 반복 
# n_required_iterations: 4      / 필수 반복
# n_possible_iterations: 4      / 가능한 반복
# min_resources_: 179           / 최소 리소스 
# max_resources_: 1437          / 최대 리소스 는 최소에 x2, x2, x2 한거임.
# aggressive_elimination: False / 공격적인 제거
# factor: 2                     / 인자

# ----------
# iter: 0                       / 아이터
# n_candidates: 9               / 후보 / 2번째 돌릴때는 9개에서 5개로 줄어듬.
# n_resources: 179              / 자원 / 2번째는 358로 많아짐. 2배 늘어남. 펙터가 2라서 x2한거임.

# iter: 1
# n_candidates: 5
# n_resources: 358
# Fitting 5 folds for each of 5 candidates, totalling 25 fits

# iter: 2
# n_candidates: 3
# n_resources: 716
# Fitting 5 folds for each of 3 candidates, totalling 15 fits

# iter: 3
# n_candidates: 2               / 좋은 것 2개 중에서 1개를 얻기 위해서
# n_resources: 1432             / 전체 데이터를 넣어서 돌림.
# Fitting 5 folds for each of 2 candidates, totalling 10 fits


# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device='cuda', early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.1, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=3, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=50, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...)
# 최적의 파라미터 {'learning_rate': 0.1, 'max_depth': 3} <- 그래서 
# best_score :  0.9482517482517482
# model.score : 0.9444444444444444
# accuracy_score : 0.9444444444444444
# accuracy_score : 0.9444444444444444
# 걸린 시간 : 513.62 초

# ==================================================================================
# factor=3       
# n_iterations: 3               / 반복 
# n_required_iterations: 3      / 필수 반복
# n_possible_iterations: 3      / 가능한 반복
# min_resources_: 159           / 최소 리소스 
# max_resources_: 1437          / 최대 리소스 는 최소에 x2, x2, x2 한거임.
# aggressive_elimination: False / 공격적인 제거
# factor: 3                     / 인자

# ----------
# iter: 0
# n_candidates: 9
# n_resources: 159

# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device='cuda', early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.1, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=3, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=50, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...)
# 최적의 파라미터 {'learning_rate': 0.1, 'max_depth': 3}
# best_score :  0.9474420316525579
# model.score : 0.9444444444444444
# accuracy_score : 0.9444444444444444
# accuracy_score : 0.9444444444444444
# 걸린 시간 : 308.04 초

###  min_resources= 100, max_resources= 1437, ###
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device='cuda', early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.05, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=4, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=50, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...)
# 최적의 파라미터 {'learning_rate': 0.05, 'max_depth': 4}
# best_score :  0.9186468032278088
# model.score : 0.9361111111111111
# accuracy_score : 0.9361111111111111
# accuracy_score : 0.9361111111111111
# 걸린 시간 : 275.27 초

### min_resources= 30, max_resources= 1437, aggressive_elimination=True,###
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=0.7, device='cuda', early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.2, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=None, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=50, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...)
# 최적의 파라미터 {'colsample_bytree': 0.7, 'learning_rate': 0.2}
# best_score :  0.953153320918684
# model.score : 0.9666666666666667
# accuracy_score : 0.9666666666666667
# accuracy_score : 0.9666666666666667
# 걸린 시간 : 2805.21 초
