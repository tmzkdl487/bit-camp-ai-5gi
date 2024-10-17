# 뭐하래는건지 알것지?

from sklearn.datasets import fetch_california_housing

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# 1. 데이터
datasets = fetch_california_housing()
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
# Original R2 Score: 0.5964140465722068
# Original Feature Importances: [0.51873533 0.05014494 0.05060456 0.02551158 0.02781676 0.13387334
#  0.09833673 0.09497676]
# R2 Score after removing 2 features and adding PCA component: 0.6311599029544155

# ================= RandomForestRegressor =================
# Original R2 Score: 0.811439104037621
# Original Feature Importances: [0.52445075 0.05007899 0.04596161 0.03031591 0.03121773 0.1362301
#  0.09138102 0.09036389]
# R2 Score after removing 2 features and adding PCA component: 0.8143977382581551

# ================= GradientBoostingRegressor =================
# Original R2 Score: 0.7865333436969877
# Original Feature Importances: [0.60051609 0.02978481 0.02084099 0.00454408 0.0027597  0.12535772
#  0.08997582 0.12622079]
# R2 Score after removing 2 features and adding PCA component: 0.7892350927997399

# ================= XGBRegressor =================
# Original R2 Score: 0.8384930657222394
# Original Feature Importances: [0.49375907 0.06520814 0.04559402 0.02538511 0.02146595 0.14413244
#  0.0975963  0.10685894]
# R2 Score after removing 2 features and adding PCA component: 0.8396699203549337