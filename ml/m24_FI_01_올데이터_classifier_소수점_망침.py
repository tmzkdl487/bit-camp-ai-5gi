from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

import numpy as np

#1. 데이터셋 로드
datasets = {
    "06_cancer": load_breast_cancer(return_X_y=True),
    "09_wine": load_wine(return_X_y=True),
    "11_digits": load_digits(return_X_y=True)
}

random_state = 1223

#2. 모델 리스트 정의
models = [
    DecisionTreeClassifier(random_state=random_state),
    RandomForestClassifier(random_state=random_state),
    GradientBoostingClassifier(random_state=random_state),
    XGBClassifier(random_state=random_state)
]

#3. 각 데이터셋에 대해 모델을 학습 및 평가
for name, (x, y) in datasets.items():
    print(f"### Dataset: {name} ###")
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                        random_state=random_state, 
                                                        stratify=y if len(np.unique(y)) > 1 else None)
    
    for model in models:
        model.fit(x_train, y_train)
        print("===================", model.__class__.__name__, "=====================")
        print('acc', model.score(x_test, y_test))
        
        # 특성 중요도를 소수점 2자리로 포맷팅하여 출력
        feature_importances = np.round(model.feature_importances_, 2)
        
        # 특성 중요도가 길면 첫 10개만 출력하고 "..."으로 생략
        if len(feature_importances) > 10:
            print(f'feature importances: {feature_importances[:10]} ...')
        else:
            print(f'feature importances: {feature_importances}')
    print("\n")  # 각 데이터셋의 결과를 구분하기 위해 추가
    
# ### Dataset: 06_cancer ###
# =================== DecisionTreeClassifier =====================
# acc 0.9473684210526315
# feature importances: [0.   0.05 0.   0.   0.   0.   0.   0.   0.   0.01] ...
# =================== RandomForestClassifier =====================
# acc 0.9298245614035088
# feature importances: [0.02 0.01 0.04 0.06 0.   0.03 0.03 0.07 0.   0.  ] ...
# =================== GradientBoostingClassifier =====================
# acc 0.9385964912280702
# feature importances: [0.   0.02 0.   0.   0.   0.   0.01 0.05 0.   0.  ] ...
# =================== XGBClassifier =====================
# acc 0.9385964912280702
# feature importances: [0.01 0.02 0.   0.   0.   0.01 0.   0.05 0.   0.  ] ...


# ### Dataset: 09_wine ###
# =================== DecisionTreeClassifier =====================
# acc 0.8611111111111112
# feature importances: [0.02 0.   0.   0.04 0.   0.   0.14 0.   0.   0.  ] ...
# =================== RandomForestClassifier =====================
# acc 0.9444444444444444
# feature importances: [0.14 0.02 0.01 0.04 0.03 0.05 0.14 0.01 0.03 0.14] ...
# =================== GradientBoostingClassifier =====================
# acc 0.9166666666666666
# feature importances: [0.14 0.04 0.01 0.   0.01 0.   0.11 0.   0.   0.16] ...
# =================== XGBClassifier =====================
# acc 0.9444444444444444
# feature importances: [0.08 0.03 0.04 0.   0.01 0.   0.08 0.   0.03 0.09] ...


# ### Dataset: 11_digits ###
# =================== DecisionTreeClassifier =====================
# acc 0.8472222222222222
# feature importances: [0.   0.01 0.01 0.02 0.01 0.05 0.   0.   0.   0.02] ...
# =================== RandomForestClassifier =====================
# acc 0.9777777777777777
# feature importances: [0.   0.   0.02 0.01 0.01 0.02 0.01 0.   0.   0.01] ...
# =================== GradientBoostingClassifier =====================
# acc 0.9638888888888889
# feature importances: [0.   0.   0.01 0.   0.   0.06 0.01 0.   0.   0.  ] ...
# =================== XGBClassifier =====================
# acc 0.9694444444444444
# feature importances: [0.   0.03 0.01 0.01 0.01 0.04 0.01 0.   0.   0.01] ...