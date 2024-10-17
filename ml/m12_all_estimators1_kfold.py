# 얘는 리그레서로 맹그러

from sklearn.datasets import load_iris

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators    # all_estimators 모든 측정기

import sklearn as sk

import warnings
warnings.filterwarnings('ignore')   # 워닝 무시

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8, stratify=y,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
all = all_estimators(type_filter='classifier') 

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

for name, model in all:
    try:
        #2. 모델 
        model = model()
        
        #3. 훈련
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print("========================", name, "=================")
        print("ACC : ", scores, '\n, 평균ACC : ', round(np.mean(scores), 4))
        
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predict)
        print('cross_val_predict ACC : ', acc)
        
    except:
        print(name, '은 바보 멍충이!!!')

# ======================== AdaBoostClassifier =================   
# ACC :  [0.91666667 0.95833333 0.95833333 0.95833333 0.875     ] 
# , 평균ACC :  0.9333
# cross_val_predict ACC :  0.9666666666666667
# ======================== BaggingClassifier =================    
# ACC :  [0.91666667 0.91666667 1.         0.95833333 0.91666667] 
# , 평균ACC :  0.9417
# cross_val_predict ACC :  0.9666666666666667
# ======================== BernoulliNB =================
# ACC :  [0.70833333 0.875      0.875      0.75       0.66666667]
# , 평균ACC :  0.775
# cross_val_predict ACC :  0.5666666666666667
# ======================== CalibratedClassifierCV =================
# ACC :  [0.875      0.91666667 0.91666667 0.875      0.875     ]
# , 평균ACC :  0.8917
# cross_val_predict ACC :  0.8
# CategoricalNB 은 바보 멍충이!!!
# ClassifierChain 은 바보 멍충이!!!
# ComplementNB 은 바보 멍충이!!!
# ======================== DecisionTreeClassifier =================
# ACC :  [0.91666667 0.91666667 1.         0.95833333 0.91666667]
# , 평균ACC :  0.9417
# cross_val_predict ACC :  0.9666666666666667
# ======================== DummyClassifier =================
# ACC :  [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333]
# , 평균ACC :  0.3333
# cross_val_predict ACC :  0.3333333333333333
# ======================== ExtraTreeClassifier =================
# ACC :  [0.95833333 1.         1.         0.91666667 1.        ]
# , 평균ACC :  0.975
# cross_val_predict ACC :  0.9666666666666667
# ======================== ExtraTreesClassifier =================
# ACC :  [0.91666667 0.95833333 1.         1.         0.91666667]
# , 평균ACC :  0.9583
# cross_val_predict ACC :  0.9666666666666667
# ======================== GaussianNB =================
# ACC :  [0.91666667 0.95833333 1.         0.95833333 0.91666667]
# , 평균ACC :  0.95
# cross_val_predict ACC :  0.9333333333333333
# ======================== GaussianProcessClassifier =================
# ACC :  [0.91666667 0.95833333 1.         0.91666667 0.91666667]
# , 평균ACC :  0.9417
# cross_val_predict ACC :  0.9333333333333333
# ======================== GradientBoostingClassifier =================
# ACC :  [0.91666667 0.91666667 0.95833333 0.95833333 0.91666667]
# , 평균ACC :  0.9333
# cross_val_predict ACC :  0.9333333333333333
# ======================== HistGradientBoostingClassifier =================
# ACC :  [0.91666667 1.         0.95833333 0.95833333 0.91666667]
# , 평균ACC :  0.95
# cross_val_predict ACC :  0.3333333333333333
# ======================== KNeighborsClassifier =================
# ACC :  [0.95833333 0.95833333 1.         1.         0.875     ]
# , 평균ACC :  0.9583
# cross_val_predict ACC :  0.9666666666666667
# ======================== LabelPropagation =================
# ACC :  [0.91666667 0.875      1.         0.95833333 0.83333333]
# , 평균ACC :  0.9167
# cross_val_predict ACC :  0.8
# ======================== LabelSpreading =================
# ACC :  [0.91666667 0.875      1.         0.95833333 0.83333333]
# , 평균ACC :  0.9167
# cross_val_predict ACC :  0.8
# ======================== LinearDiscriminantAnalysis =================
# ACC :  [0.95833333 0.95833333 1.         1.         0.95833333]
# , 평균ACC :  0.975
# cross_val_predict ACC :  1.0
# ======================== LinearSVC =================
# ACC :  [0.91666667 0.91666667 1.         0.91666667 0.91666667]
# , 평균ACC :  0.9333
# cross_val_predict ACC :  0.8333333333333334
# ======================== LogisticRegression =================
# ACC :  [0.91666667 0.95833333 1.         1.         0.91666667]
# , 평균ACC :  0.9583
# cross_val_predict ACC :  0.9
# ======================== LogisticRegressionCV =================
# ACC :  [0.95833333 0.95833333 1.         1.         0.91666667]
# , 평균ACC :  0.9667
# cross_val_predict ACC :  0.9333333333333333
# ======================== MLPClassifier =================
# ACC :  [0.91666667 0.95833333 1.         1.         0.91666667]
# , 평균ACC :  0.9583
# cross_val_predict ACC :  0.9
# MultiOutputClassifier 은 바보 멍충이!!!
# MultinomialNB 은 바보 멍충이!!!
# ======================== NearestCentroid =================
# ACC :  [0.83333333 0.95833333 0.91666667 0.875      0.83333333]
# , 평균ACC :  0.8833
# cross_val_predict ACC :  0.7666666666666667
# ======================== NuSVC =================
# ACC :  [0.95833333 0.95833333 1.         1.         0.91666667]
# , 평균ACC :  0.9667
# cross_val_predict ACC :  0.9
# OneVsOneClassifier 은 바보 멍충이!!!
# OneVsRestClassifier 은 바보 멍충이!!!
# OutputCodeClassifier 은 바보 멍충이!!!
# ======================== PassiveAggressiveClassifier =================
# ACC :  [0.91666667 0.79166667 1.         0.91666667 0.79166667]
# , 평균ACC :  0.8833
# cross_val_predict ACC :  0.8333333333333334
# ======================== Perceptron =================
# ACC :  [0.875      0.83333333 1.         0.875      0.95833333]
# , 평균ACC :  0.9083
# cross_val_predict ACC :  0.7666666666666667
# ======================== QuadraticDiscriminantAnalysis =================
# ACC :  [0.95833333 0.95833333 1.         1.         0.95833333]
# , 평균ACC :  0.975
# cross_val_predict ACC :  0.9666666666666667
# ======================== RadiusNeighborsClassifier =================
# ACC :  [0.91666667        nan        nan 1.         0.875     ]
# , 평균ACC :  nan
# RadiusNeighborsClassifier 은 바보 멍충이!!!
# ======================== RandomForestClassifier =================
# ACC :  [0.91666667 0.91666667 1.         0.95833333 0.91666667]
# , 평균ACC :  0.9417
# cross_val_predict ACC :  0.9666666666666667
# ======================== RidgeClassifier =================
# ACC :  [0.79166667 0.83333333 0.91666667 0.875      0.75      ]
# , 평균ACC :  0.8333
# cross_val_predict ACC :  0.8
# ======================== RidgeClassifierCV =================
# ACC :  [0.79166667 0.875      0.91666667 0.875      0.75      ]
# , 평균ACC :  0.8417
# cross_val_predict ACC :  0.8
# ======================== SGDClassifier =================
# ACC :  [0.91666667 0.875      1.         0.875      0.91666667]
# , 평균ACC :  0.9167
# cross_val_predict ACC :  0.8
# ======================== SVC =================
# ACC :  [0.95833333 0.95833333 1.         1.         0.91666667]
# , 평균ACC :  0.9667
# cross_val_predict ACC :  0.8666666666666667
# StackingClassifier 은 바보 멍충이!!!
# VotingClassifier 은 바보 멍충이!!!


