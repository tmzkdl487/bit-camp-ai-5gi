import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_digits(return_X_y=True)

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
    metric_name='mlogloss', # 이진: logloss, 다중: mlogloss, error
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

thresholds = np.sort(model.feature_importances_)    # 오름차순.
print(thresholds)

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

# 최종점수 :  0.9777777777777777
# accuracy_score :  0.9777777777777777
# [0.00000000e+00 7.22037116e-03 9.22979973e-03 7.61816511e-03
#  1.00602135e-02 3.40358689e-02 5.73287299e-03 1.63839001e-03
#  6.02039800e-05 8.09979998e-03 2.24853233e-02 3.91543005e-03
#  1.04004694e-02 1.36420894e-02 8.39948002e-03 1.08191081e-04
#  5.30562465e-05 7.22476281e-03 1.38617987e-02 3.00309639e-02
#  1.58374645e-02 5.67955077e-02 1.24280313e-02 7.74079526e-05
#  3.51197203e-04 8.90468154e-03 3.17197479e-02 1.97219830e-02
#  2.00204011e-02 2.96405852e-02 1.37459971e-02 1.80533389e-04
#  0.00000000e+00 4.97237407e-02 1.84153263e-02 2.06911135e-02
#  5.92775755e-02 3.00222747e-02 2.23511942e-02 0.00000000e+00
#  3.17727881e-06 6.16753241e-03 3.71117517e-02 4.71289046e-02
#  1.51885599e-02 1.10755926e-02 1.69138443e-02 2.87044910e-04
#  9.16274264e-03 2.06111372e-03 1.13352640e-02 1.68603808e-02
#  5.68483863e-03 2.17179377e-02 3.70346494e-02 9.53246467e-03
#  0.00000000e+00 3.72822420e-03 2.97947563e-02 5.24723437e-03
#  5.37617952e-02 1.73609722e-02 2.48197839e-02 1.42993852e-02]
# [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#  3.17727881e-06 5.30562465e-05 6.02039800e-05 7.74079526e-05
#  1.08191081e-04 1.80533389e-04 2.87044910e-04 3.51197203e-04
#  1.63839001e-03 2.06111372e-03 3.72822420e-03 3.91543005e-03
#  5.24723437e-03 5.68483863e-03 5.73287299e-03 6.16753241e-03
#  7.22037116e-03 7.22476281e-03 7.61816511e-03 8.09979998e-03
#  8.39948002e-03 8.90468154e-03 9.16274264e-03 9.22979973e-03
#  9.53246467e-03 1.00602135e-02 1.04004694e-02 1.10755926e-02
#  1.13352640e-02 1.24280313e-02 1.36420894e-02 1.37459971e-02
#  1.38617987e-02 1.42993852e-02 1.51885599e-02 1.58374645e-02
#  1.68603808e-02 1.69138443e-02 1.73609722e-02 1.84153263e-02
#  1.97219830e-02 2.00204011e-02 2.06911135e-02 2.17179377e-02
#  2.23511942e-02 2.24853233e-02 2.48197839e-02 2.96405852e-02
#  2.97947563e-02 3.00222747e-02 3.00309639e-02 3.17197479e-02
#  3.40358689e-02 3.70346494e-02 3.71117517e-02 4.71289046e-02
#  4.97237407e-02 5.37617952e-02 5.67955077e-02 5.92775755e-02]
# Trech=0.000, n=64, ACC:97.78%
# Trech=0.000, n=64, ACC:97.78%
# Trech=0.000, n=64, ACC:97.78%
# Trech=0.000, n=64, ACC:97.78%
# Trech=0.000, n=60, ACC:97.78%
# Trech=0.000, n=59, ACC:97.78%
# Trech=0.000, n=58, ACC:97.78%
# Trech=0.000, n=57, ACC:97.78%
# Trech=0.000, n=56, ACC:97.78%
# Trech=0.000, n=55, ACC:97.78%
# Trech=0.000, n=54, ACC:97.78%
# Trech=0.000, n=53, ACC:97.78%
# Trech=0.002, n=52, ACC:97.78%
# Trech=0.002, n=51, ACC:97.78%
# Trech=0.004, n=50, ACC:97.50%
# Trech=0.004, n=49, ACC:97.50%
# Trech=0.005, n=48, ACC:98.33%
# Trech=0.006, n=47, ACC:97.78%
# Trech=0.006, n=46, ACC:97.78%
# Trech=0.006, n=45, ACC:97.78%
# Trech=0.007, n=44, ACC:97.22%
# Trech=0.007, n=43, ACC:97.78%
# Trech=0.008, n=42, ACC:97.50%
# Trech=0.008, n=41, ACC:97.78%
# Trech=0.008, n=40, ACC:97.78%
# Trech=0.009, n=39, ACC:97.78%
# Trech=0.009, n=38, ACC:97.78%
# Trech=0.009, n=37, ACC:97.78%
# Trech=0.010, n=36, ACC:97.78%
# Trech=0.010, n=35, ACC:97.78%
# Trech=0.010, n=34, ACC:97.50%
# Trech=0.011, n=33, ACC:97.78%
# Trech=0.011, n=32, ACC:97.78%
# Trech=0.012, n=31, ACC:97.50%
# Trech=0.014, n=30, ACC:97.22%
# Trech=0.014, n=29, ACC:97.22%
# Trech=0.014, n=28, ACC:96.67%
# Trech=0.014, n=27, ACC:96.67%
# Trech=0.015, n=26, ACC:96.39%
# Trech=0.016, n=25, ACC:95.83%
# Trech=0.017, n=24, ACC:96.11%
# Trech=0.017, n=23, ACC:96.67%
# Trech=0.017, n=22, ACC:96.11%
# Trech=0.018, n=21, ACC:95.83%
# Trech=0.020, n=20, ACC:95.28%
# Trech=0.020, n=19, ACC:96.11%
# Trech=0.021, n=18, ACC:95.28%
# Trech=0.022, n=17, ACC:95.28%
# Trech=0.022, n=16, ACC:94.72%
# Trech=0.022, n=15, ACC:95.28%
# Trech=0.025, n=14, ACC:94.44%
# Trech=0.030, n=13, ACC:95.28%
# Trech=0.030, n=12, ACC:94.72%
# Trech=0.030, n=11, ACC:93.06%
# Trech=0.030, n=10, ACC:91.94%
# Trech=0.032, n=9, ACC:90.00%
# Trech=0.034, n=8, ACC:85.00%
# Trech=0.037, n=7, ACC:83.89%
# Trech=0.037, n=6, ACC:76.94%
# Trech=0.047, n=5, ACC:71.67%
# Trech=0.050, n=4, ACC:59.17%
# Trech=0.054, n=3, ACC:44.72%
# Trech=0.057, n=2, ACC:37.78%
# Trech=0.059, n=1, ACC:25.56%
