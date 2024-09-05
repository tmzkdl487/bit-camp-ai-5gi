from sklearn.datasets import load_diabetes

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# 1. 데이터
datasets = load_diabetes()
df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
x = df.values
y = datasets.target

# 데이터 분할
random_state1 = 1223
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state1)

# 2. 모델 구성
model1 = DecisionTreeRegressor(random_state=random_state1)
model2 = RandomForestRegressor(random_state=random_state1)
model3 = GradientBoostingRegressor(random_state=random_state1)
model4 = XGBRegressor(random_state=random_state1)

models = [model1, model2, model3, model4]

for model in models:
    model.fit(x_train, y_train)
    feature_importances = model.feature_importances_
    
    # 중요도 기반 정렬 (내림차순)
    sorted_idx = np.argsort(feature_importances)
    
    print(f"\n================= {model.__class__.__name__} =================")
    print('Original R2 Score:', r2_score(y_test, model.predict(x_test)))
    print('Original Feature Importances:', feature_importances)
    
    # 하위 30% 특성 제거
    n_remove = int(len(sorted_idx) * 0.3)
    removed_features_idx = sorted_idx[:n_remove]  # 하위 30% 특성 제거
    
    # 제거된 특성들
    x_train_removed = x_train[:, removed_features_idx]
    x_test_removed = x_test[:, removed_features_idx]
    
    # PCA를 사용해 제거된 특성들을 병합
    pca = PCA(n_components=1)  # 주성분 1개로 변환
    x_train_pca = pca.fit_transform(x_train_removed)
    x_test_pca = pca.transform(x_test_removed)
    
    # PCA로 변환된 특성을 원래 데이터셋에 추가
    x_train_augmented = np.hstack((np.delete(x_train, removed_features_idx, axis=1), x_train_pca))
    x_test_augmented = np.hstack((np.delete(x_test, removed_features_idx, axis=1), x_test_pca))
    
    # 모델 재학습 및 평가
    model.fit(x_train_augmented, y_train)
    r2_reduced = r2_score(y_test, model.predict(x_test_augmented))
    
    print(f"R2 Score after removing {n_remove} features and adding PCA component: {r2_reduced}")

# ================= DecisionTreeRegressor =================
# Original R2 Score: -0.24733855513252667
# Original Feature Importances: [0.05676749 0.01855931 0.23978058 0.08279462 0.05873671 0.0639961
#  0.04130515 0.01340568 0.33217096 0.0924834 ]
# R2 Score after removing 3 features and adding PCA component: -0.2585747642735843

# ================= RandomForestRegressor =================
# Original R2 Score: 0.3687286985683689
# Original Feature Importances: [0.05394197 0.00931513 0.25953258 0.1125408  0.04297661 0.05293764
#  0.06684433 0.02490964 0.29157054 0.08543076]
# R2 Score after removing 3 features and adding PCA component: 0.3414858670900289

# ================= GradientBoostingRegressor =================
# Original R2 Score: 0.3647974813076822
# Original Feature Importances: [0.04509096 0.00780692 0.25858035 0.09953666 0.02605597 0.06202725
#  0.05303144 0.01840481 0.35346141 0.07600423]
# R2 Score after removing 3 features and adding PCA component: 0.31852523231456864

# ================= XGBRegressor =================
# Original R2 Score: 0.10076704957922011
# Original Feature Importances: [0.04070464 0.0605858  0.16995801 0.06239288 0.06619858 0.06474677
#  0.05363544 0.03795785 0.35376146 0.09005855]
# R2 Score after removing 3 features and adding PCA component: 0.22999499763454545