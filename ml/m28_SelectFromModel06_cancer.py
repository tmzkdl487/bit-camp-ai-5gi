import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
# print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, 
                                                    random_state= 3377, 
                                                    stratify=y,
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import xgboost as xgb
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name='logloss', # 이진: logloss, 다중: mlogloss, error
    data_name='validation_0',
    # save_best=True,   # ??????? 이거 넣으면 AttributeError: `best_iteration` is only defined when early stopping is used. 에러남.
)

#2. 모델
model = XGBClassifier(
    n_estimators = 1000,
    max_depth = 8,
    gamma = 0,
    min_child_weight= 0,
    subsample= 0.4,
    reg_alpha= 0,    # L1 규제 리소
    reg_lambda= 1,   # L2 규제 라지
    # eval_metric='mlogloss', # 2.1.1 버전에서 컴파일이 아니라 모델로 가야됨.. / 이진분류에서error도 쓸 수 있음.
    callbacks=[early_stop], 
    random_state=3377,
)

#3. 컴파일, 훈련
model.fit(x_train, y_train,
         eval_set=[(x_test, y_test)],
         verbose=1,
        #   eval_metric='mlogloss', # 2.1.1 버전에서 위로감.
          )

#4. 평가, 예측
results =  model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

print(model.feature_importances_)
# [3.0102523e-03 2.5315402e-02 4.0497244e-03 1.6470300e-05 2.0678829e-02
#  8.2311193e-03 5.0021070e-03 8.1781179e-02 6.8232184e-03 5.5803633e-03
#  1.1685277e-02 1.4600820e-03 5.5390969e-03 6.1498541e-02 8.1495577e-03
#  6.0727349e-03 7.5648869e-03 1.1760910e-02 1.5135714e-02 5.4912721e-03
#  5.3025778e-02 5.0092224e-02 6.6685311e-02 2.3978993e-01 2.3762172e-02
#  1.8026562e-02 2.8897034e-02 1.8585879e-01 1.6002852e-02 2.3012649e-02]

thresholds = np.sort(model.feature_importances_)    # 오름차순.
print(thresholds)

# 최종점수 :  최종점수 :  0.9666666666666667 / accuracy_score :  0.9666666666666667

# 최종점수 :  0.9912280701754386 / accuracy_score :  0.9912280701754386

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    select_model = XGBClassifier(  n_estimators = 1000,
                                    max_depth = 8,
                                    gamma = 0,
                                    min_child_weight= 0,
                                    subsample= 0.4,
                                    # reg_alpha= 0,    # L1 규제 리소
                                    # reg_lambda= 1,   # L2 규제 라지
                                    # eval_metric='mlogloss', # 2.1.1 버전에서 컴파일이 아니라 모델로 가야됨.. / 이진분류에서error도 쓸 수 있음.
                                    # callbacks=[early_stop], 
                                    random_state=3377,)
    
    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_test, y_test)],
                     verbose=0,
                     )
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    
    print('Trech=%.3f, n=%d, ACC:%.2f%%' %(i, select_x_train.shape[1], score*100))

# callbacks=[early_stop] 넣었을 때.
# Trech=0.002, n=30, ACC:86.84%
# Trech=0.003, n=29, ACC:86.84%
# Trech=0.004, n=28, ACC:86.84%
# Trech=0.004, n=27, ACC:86.84%
# Trech=0.005, n=26, ACC:86.84%
# Trech=0.005, n=25, ACC:86.84%
# Trech=0.006, n=24, ACC:86.84%
# Trech=0.006, n=23, ACC:86.84%
# Trech=0.007, n=22, ACC:86.84%
# Trech=0.007, n=21, ACC:86.84%
# Trech=0.007, n=20, ACC:86.84%
# Trech=0.008, n=19, ACC:86.84%
# Trech=0.009, n=18, ACC:86.84%
# Trech=0.011, n=17, ACC:86.84%
# Trech=0.011, n=16, ACC:86.84%
# Trech=0.013, n=15, ACC:86.84%
# Trech=0.014, n=14, ACC:86.84%
# Trech=0.016, n=13, ACC:86.84%
# Trech=0.021, n=12, ACC:86.84%
# Trech=0.022, n=11, ACC:86.84%
# Trech=0.023, n=10, ACC:87.72%
# Trech=0.027, n=9, ACC:87.72%
# Trech=0.031, n=8, ACC:87.72%
# Trech=0.049, n=7, ACC:87.72%
# Trech=0.056, n=6, ACC:87.72%
# Trech=0.061, n=5, ACC:91.23%
# Trech=0.066, n=4, ACC:91.23%
# Trech=0.079, n=3, ACC:92.98%
# Trech=0.186, n=2, ACC:91.23%
# Trech=0.244, n=1, ACC:92.98%

# callbacks=[early_stop] 안 넣었을 때.
# Trech=0.002, n=30, ACC:99.12% n은 열의 갯수
# Trech=0.003, n=29, ACC:99.12%
# Trech=0.004, n=28, ACC:99.12%
# Trech=0.004, n=27, ACC:99.12%
# Trech=0.005, n=26, ACC:99.12%
# Trech=0.005, n=25, ACC:99.12%
# Trech=0.006, n=24, ACC:99.12%
# Trech=0.006, n=23, ACC:99.12%
# Trech=0.007, n=22, ACC:99.12%
# Trech=0.007, n=21, ACC:99.12%
# Trech=0.007, n=20, ACC:99.12%
# Trech=0.008, n=19, ACC:99.12%
# Trech=0.009, n=18, ACC:99.12%
# Trech=0.011, n=17, ACC:99.12%
# Trech=0.011, n=16, ACC:99.12%
# Trech=0.013, n=15, ACC:98.25%
# Trech=0.014, n=14, ACC:98.25%
# Trech=0.016, n=13, ACC:99.12%
# Trech=0.021, n=12, ACC:99.12% <- 여기까지 성능이 같음.
# Trech=0.022, n=11, ACC:98.25%
# Trech=0.023, n=10, ACC:98.25%
# Trech=0.027, n=9, ACC:96.49%
# Trech=0.031, n=8, ACC:97.37%
# Trech=0.049, n=7, ACC:96.49%
# Trech=0.056, n=6, ACC:97.37%
# Trech=0.061, n=5, ACC:94.74%
# Trech=0.066, n=4, ACC:92.98%
# Trech=0.079, n=3, ACC:94.74%
# Trech=0.186, n=2, ACC:94.74%
# Trech=0.244, n=1, ACC:92.11% 
