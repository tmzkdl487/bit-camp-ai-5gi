import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV 
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

import xgboost as xgb
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name='mlogloss', # 또는 error
    data_name='validation_0',
    save_best=True,
)

#2. 모델
model = HalvingRandomSearchCV(XGBClassifier(
                                          # tree_method='gpu_hist'
                                          tree_method='hist',
                                          device='cuda',
                                          n_estimators=500,
                                          eval_metric='mlogloss',
                                          callbacks=[early_stop] 
                                          ),
                     parameters, 
                     cv=kfold,
                     verbose=1, # 1이 아니고 2나 3이 좋아.
                     refit=True,
                    #  n_jobs=-1,
                    #  n_iter=10,
                     random_state=4325,
                     factor= 3,   #  HalvingGridSearchCV는 성능이 가장 낮은 절반의 조합을 제거합니다. factor=2이므로 절반으로 줄어듭니다.
                     min_resources= 30,
                     max_resources= 1437,
                     aggressive_elimination=True,
                     )

#3. 훈련
start_time = time.time()
model.fit(x_train,y_train,
          verbose=False,
          eval_set=[(x_test, y_test)],  # 발리데이션 기준임.
          )
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

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 30
# max_resources_: 1437
# aggressive_elimination: True
# factor: 3

# ----------
# iter: 0
# n_candidates: 47
# n_resources: 30
# Fitting 5 folds for each of 47 candidates, totalling 235 fits

