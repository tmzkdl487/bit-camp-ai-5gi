# from sklearn.datasets import fetch_california_housing

# import pandas as pd
# import numpy as np

# from sklearn.preprocessing import MinMaxScaler
# from xgboost import XGBClassifier
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, r2_score

# # 1. 데이터셋 로드
# iris = fetch_california_housing()
# x, y = iris.data, iris.target

# feature_names = iris.feature_names

# # 2. 데이터프레임으로 변환
# df = pd.DataFrame(x, columns=feature_names)

# # 3. 데이터 분할
# x_train, x_test, y_train, y_test = train_test_split(
#     df, y, train_size=0.8, shuffle=True, random_state=337, 
#     # stratify=y
# )

# # 4. 데이터 스케일링
# scaler = MinMaxScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# # 5. 모델 정의 및 훈련
# model = XGBRegressor(random_state=337)
# model.fit(x_train_scaled, y_train)

# # 6. 예측 및 정확도 평가
# y_predict = model.predict(x_test_scaled)
# r2 = r2_score(y_test, y_predict)
# print(f"======== XGBRegressor ==")
# print("Original r2_score: ", r2)

# # 7. 피처 임포턴스 추출 및 하위 20~25% 컬럼 제거
# feature_importances = model.feature_importances_
# threshold = np.percentile(feature_importances, 25)  # 하위 25% 임계값

# # 중요도가 낮은 컬럼을 필터링
# low_importance_features = [feature for feature, importance in zip(df.columns, feature_importances) if importance <= threshold]

# # 컬럼 제거 후 데이터셋 재구성
# df_reduced = df.drop(columns=low_importance_features)

# # 8. 데이터 분할 및 스케일링 (재구성된 데이터셋)
# x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
#     df_reduced, y, train_size=0.8, shuffle=True, random_state=337, 
#     # stratify=y
# )
# x_train_reduced_scaled = scaler.fit_transform(x_train_reduced)
# x_test_reduced_scaled = scaler.transform(x_test_reduced)

# # 9. 재구성된 데이터셋으로 모델 재훈련 및 평가
# model.fit(x_train_reduced_scaled, y_train_reduced)
# y_predict_reduced = model.predict(x_test_reduced_scaled)
# # acc_reduced = accuracy_score(y_test_reduced, y_predict_reduced)

# r2_reduced  = r2_score(y_test_reduced, y_predict_reduced)

# # 10. 결과 비교
# print(f"Reduced r2_score: {r2_reduced}")
# print(f"Removed features: {low_importance_features}")

# ======== XGBRegressor ==
# Original r2_score:  0.8272149321242179
# Reduced r2_score: 0.83390894774375
# Removed features: ['AveBedrms', 'Population']

from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
    
    # 하위 10%, 20%, 30%, 40%, 50% 제거하고 R2 스코어 계산
    for percentage in [10, 20, 30, 40, 50]:
        n_remove = int(len(sorted_idx) * (percentage / 100))
        removed_features_idx = sorted_idx[:n_remove]  # 하위 n% 특성 제거
        
        # 제거된 특성을 제외한 데이터셋 생성
        x_train_reduced = np.delete(x_train, removed_features_idx, axis=1)
        x_test_reduced = np.delete(x_test, removed_features_idx, axis=1)
        
        # 모델 재학습 및 평가
        model.fit(x_train_reduced, y_train)
        r2_reduced = r2_score(y_test, model.predict(x_test_reduced))
        
        print(f"R2 Score after removing {percentage}% lowest importance features: {r2_reduced}")

# ================= DecisionTreeRegressor =================
# Original R2 Score: 0.5964140465722068
# Original Feature Importances: [0.51873533 0.05014494 0.05060456 0.02551158 0.02781676 0.13387334
#  0.09833673 0.09497676]
# R2 Score after removing 10% lowest importance features: 0.5964140465722068
# R2 Score after removing 20% lowest importance features: 0.6305064600941417
# R2 Score after removing 30% lowest importance features: 0.6180060968649153
# R2 Score after removing 40% lowest importance features: 0.6680538555842082
# R2 Score after removing 50% lowest importance features: 0.6342429706882687

# ================= RandomForestRegressor =================
# Original R2 Score: 0.811439104037621
# Original Feature Importances: [0.52445075 0.05007899 0.04596161 0.03031591 0.03121773 0.1362301
#  0.09138102 0.09036389]
# R2 Score after removing 10% lowest importance features: 0.811439104037621
# R2 Score after removing 20% lowest importance features: 0.8140742736762953
# R2 Score after removing 30% lowest importance features: 0.8181229130749348
# R2 Score after removing 40% lowest importance features: 0.8157918124804502
# R2 Score after removing 50% lowest importance features: 0.8167431458461135

# ================= GradientBoostingRegressor =================
# Original R2 Score: 0.7865333436969877
# Original Feature Importances: [0.60051609 0.02978481 0.02084099 0.00454408 0.0027597  0.12535772
#  0.08997582 0.12622079]
# R2 Score after removing 10% lowest importance features: 0.7865333436969877
# R2 Score after removing 20% lowest importance features: 0.7889074157560185
# R2 Score after removing 30% lowest importance features: 0.7873932309232046
# R2 Score after removing 40% lowest importance features: 0.7902421865243833
# R2 Score after removing 50% lowest importance features: 0.7845040569867009

# ================= XGBRegressor =================
# Original R2 Score: 0.8384930657222394
# Original Feature Importances: [0.49375907 0.06520814 0.04559402 0.02538511 0.02146595 0.14413244
#  0.0975963  0.10685894]
# R2 Score after removing 10% lowest importance features: 0.8384930657222394
# R2 Score after removing 20% lowest importance features: 0.8400599862792199
# R2 Score after removing 30% lowest importance features: 0.8424404298528521
# R2 Score after removing 40% lowest importance features: 0.8396766344845554
# R2 Score after removing 50% lowest importance features: 0.836389259246098