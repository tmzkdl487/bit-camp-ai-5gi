# m15_GridSearchCV_00.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=3333, stratify=y
)


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)

parameters = [
    {"C":[1, 10, 100, 1000], 'kernel':['linear', 'sigmoid'], 'degree':[3, 4, 5]},   # 이거를 1번 돌려줘   24번 돌림
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},   # 6번 돌림
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'],
     'gamma':[0.01, 0.001, 0.0001], 'degree':[3,4]}   # 24번 돌림    
]   #

#2. 모델
model = GridSearchCV(SVC(), parameters, cv=kfold, 
                     verbose=1,
                     refit=True,    # 제일 좋은 모델 한번더 돌림.
                     n_jobs=-1, # CPU모든 코어 다 쓰라고 하는 것.
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
print('accuracy_score : ', accuracy_score(y_test, y_predict))
# 모델이 예측한 결과(y_predict)와 실제 결과(y_test)를 비교해서 
# 정확도를 계산하고 보여줍니다.

y_pred_best = model.best_estimator_.predict(x_test) # 이게 제일 나음.
print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))
# 최적의 파라미터로 설정된 모델의 정확도를 계산하고 보여줍니다. 
# 이는 최적화된 모델의 성능을 보여줍니다.

print('걸린시간: ', round(end_time - start_time,2), '초')

# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터:  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score :  nan
# model.score :  1.0
# accuracy_score :  1.0
# 최적 튠 ACC :  1.0
# 걸린시간:  1.11 초

import pandas as pd

# print(pd.DataFrame(model.cv_results_))  # 보기가 복잡하니까.
#     mean_fit_time  std_fit_time  mean_score_time  std_score_time param_C  ... split3_test_score split4_test_score mean_test_score std_test_score  rank_test_score
# 0        0.001196  3.988744e-04         0.000997    3.234067e-07       1  ...          1.000000          0.958333        0.983333       0.020412                1  
# 1        0.001794  7.455377e-04         0.000398    4.878769e-04       1  ...          0.083333          0.083333        0.075000       0.031180               43

# [54 rows x 17 columns]

# print(pd.DataFrame(model.cv_results_).T)    # .T를 넣어서 본다.
# rank_test_score                                            1

print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))

print(pd.DataFrame(model.cv_results_).columns)
# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
    #    'param_C', 'param_degree', 'param_kernel', 'param_gamma', 'params',
    #    'split0_test_score', 'split1_test_score', 'split2_test_score',
    #    'split3_test_score', 'split4_test_score', 'mean_test_score',
    #    'std_test_score', 'rank_test_score'],
    #   dtype='object')
    
path = './_save/m15_GS_CV_01/'

pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True) \
    .to_csv(path + 'm15_GS_cv_results.csv')