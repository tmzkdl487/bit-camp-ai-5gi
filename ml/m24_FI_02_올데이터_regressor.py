# 02_california
# 03_diabetse

###  요 파일에 이 2개의 데이터셋 다 넣어서 23번처럼 맹그러.

from sklearn.datasets import fetch_california_housing, load_diabetes

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRFRegressor

from sklearn.model_selection import train_test_split

#1. 데이터셋 로드
datasets = {
    "02_california": fetch_california_housing(return_X_y=True),
    "03_diabetse": load_diabetes(return_X_y=True),
}

random_state = 1223

#2. 모델 리스트 정의
models = [
    DecisionTreeRegressor(random_state=random_state),
    RandomForestRegressor(random_state=random_state),
    GradientBoostingRegressor(random_state=random_state),
    XGBRFRegressor(random_state=random_state)
]

#3. 각 데이터셋에 대해 모델을 학습 및 평가
for name, (x, y) in datasets.items():
    print(f"### Dataset: {name} ###")
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                        random_state=random_state, 
                                                        # stratify=y
                                                        )
    
    for model in models:
        model.fit(x_train, y_train)
        print("===================", model.__class__.__name__, "=====================")
        print('r2', model.score(x_test, y_test))
        print(model.feature_importances_)
    print("\n")  
    
### Dataset: 02_california ###
# =================== DecisionTreeRegressor =====================
# r2 0.5964140465722068
# [0.51873533 0.05014494 0.05060456 0.02551158 0.02781676 0.13387334
#  0.09833673 0.09497676]
# =================== RandomForestRegressor =====================
# r2 0.811439104037621
# [0.52445075 0.05007899 0.04596161 0.03031591 0.03121773 0.1362301
#  0.09138102 0.09036389]
# =================== GradientBoostingRegressor =====================
# r2 0.7865333436969877
# [0.60051609 0.02978481 0.02084099 0.00454408 0.0027597  0.12535772 
#  0.08997582 0.12622079]
# =================== XGBRFRegressor =====================
# r2 0.6973284037291707
# [0.451661   0.05509751 0.19447608 0.04169037 0.01094069 0.14598809
#  0.05534564 0.04480066]

# ### Dataset: 03_diabetse ###
# =================== DecisionTreeRegressor =====================
# r2 -0.24733855513252667
# [0.05676749 0.01855931 0.23978058 0.08279462 0.05873671 0.0639961
#  0.04130515 0.01340568 0.33217096 0.0924834 ]
# =================== RandomForestRegressor =====================
# r2 0.3687286985683689
# [0.05394197 0.00931513 0.25953258 0.1125408  0.04297661 0.05293764
#  0.06684433 0.02490964 0.29157054 0.08543076]
# =================== GradientBoostingRegressor =====================
# r2 0.3647974813076822
# [0.04509096 0.00780692 0.25858035 0.09953666 0.02605597 0.06202725
#  0.05303144 0.01840481 0.35346141 0.07600423]
# =================== XGBRFRegressor =====================
# r2 0.33589682441838786
# [0.02255717 0.02129483 0.1675215  0.06622679 0.04581491 0.05392584
#  0.05983198 0.07966285 0.39199358 0.09117053]
