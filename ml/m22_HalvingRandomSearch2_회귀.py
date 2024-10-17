from sklearn.datasets import load_diabetes
# 맹그러봐!!!!!

import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRFRegressor
import xgboost as xgb

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV 
# HalvingGridSearchCV 쓰려면은 enable_halving_search_cv을 꼭 위에 임포트해야된다.

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=15426, shuffle=True, train_size=0.8,)    # stratify=y

# print(x_train.shape, y_train.shape) # (353, 10) (353,)
# print(x_test.shape, y_test.shape)   # (89, 10) (89,)

n_splits = 5 
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    {'learning_rate' : [0.01, 0.05, 0.1,],# 0.2, 0.3], 
     'max_depth' : [3 , 4 ,5,],}, # 6, 8],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma' : [0, 0.1, 0.2, 0.5, 1.0],},
   ] # 5*5*4

#2. 모델
model = HalvingRandomSearchCV(XGBRFRegressor(
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
                     factor= 2,   #  HalvingGridSearchCV는 성능이 가장 낮은 절반의 조합을 제거합니다. factor=2이므로 절반으로 줄어듭니다.
                     min_resources= 20,
                     max_resources= 353,
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

y_predict = model.predict(x_test)

print('Mean Squared Error : ', mean_squared_error(y_test, y_predict))
print('R^2 Score : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)

print('최적 튠 ACC : ', r2_score(y_test, y_pred_best))
print('걸린 시간 :', round(end_time - start_time, 2), '초')

# 