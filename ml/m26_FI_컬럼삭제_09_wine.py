from sklearn.datasets import load_wine

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# 1. 데이터
datasets = load_wine()
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
# Original R2 Score: 0.9086294416243654
# Original Feature Importances: [0.         0.00272356 0.         0.         0.         0.00894883
#  0.62957276 0.         0.         0.12215463 0.         0.
#  0.23660022]
# R2 Score after removing 10% lowest importance features: 0.9543147208121827
# R2 Score after removing 20% lowest importance features: 0.9543147208121827
# R2 Score after removing 30% lowest importance features: 0.9543147208121827
# R2 Score after removing 40% lowest importance features: 0.9086294416243654
# R2 Score after removing 50% lowest importance features: 0.9086294416243654

# ================= RandomForestRegressor =================
# Original R2 Score: 0.9733791878172589
# Original Feature Importances: [4.12151987e-02 2.44884442e-03 2.07237534e-03 2.27620219e-03
#  6.18682263e-03 5.65400356e-04 5.26353692e-01 1.51966467e-04
#  7.53112369e-04 8.74882496e-02 4.52842520e-02 6.47519853e-02
#  2.20451899e-01]
# R2 Score after removing 10% lowest importance features: 0.9763761421319797
# R2 Score after removing 20% lowest importance features: 0.9725065989847715
# R2 Score after removing 30% lowest importance features: 0.9737994923857868
# R2 Score after removing 40% lowest importance features: 0.9703730964467006
# R2 Score after removing 50% lowest importance features: 0.9698065989847716

# ================= GradientBoostingRegressor =================
# Original R2 Score: 0.9445668767106903
# Original Feature Importances: [1.01818165e-02 1.47494583e-03 2.48264745e-05 2.44691378e-04
#  1.84084706e-03 2.19735159e-04 6.39724292e-01 5.90864801e-07
#  1.04852978e-05 1.34265116e-01 9.75678037e-04 1.39655839e-04
#  2.10897320e-01]
# R2 Score after removing 10% lowest importance features: 0.9437544906093368
# R2 Score after removing 20% lowest importance features: 0.943367233068323
# R2 Score after removing 30% lowest importance features: 0.9440790141819267
# R2 Score after removing 40% lowest importance features: 0.9450198803943214
# R2 Score after removing 50% lowest importance features: 0.9409760505846605

# ================= XGBRegressor =================
# Original R2 Score: 0.9004115776244649
# Original Feature Importances: [6.3564745e-03 7.4083154e-04 6.2418417e-07 8.8730914e-05 1.4208623e-03
#  5.3402124e-07 4.7598526e-01 5.8794353e-07 1.0890675e-06 1.1663743e-01
#  1.3028964e-03 9.7934162e-07 3.9746371e-01]
# R2 Score after removing 10% lowest importance features: 0.9004233993723969
# R2 Score after removing 20% lowest importance features: 0.900328714114792
# R2 Score after removing 30% lowest importance features: 0.9003292237072882
# R2 Score after removing 40% lowest importance features: 0.9004692753036727
# R2 Score after removing 50% lowest importance features: 0.9005919284737962        

