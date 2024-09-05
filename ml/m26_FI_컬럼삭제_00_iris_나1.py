### 판다스로 바꿔서 커럼 삭제 ###
# pd.DataFrame
# 컬럼명 :  datasets.feature_names 안에 있지!!!!

# 실습
# 피쳐임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거
# 데이터셋 재구성후
# 기존 모델결과와 비교!!!!

# 끗

from sklearn.datasets import load_iris

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터셋 로드
iris = load_iris()
x, y = iris.data, iris.target

feature_names = iris.feature_names

# 2. 데이터프레임으로 변환
df = pd.DataFrame(x, columns=feature_names)

# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    df, y, train_size=0.8, shuffle=True, random_state=337, stratify=y
)

# 4. 데이터 스케일링
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 5. 모델 정의 및 훈련
model = XGBClassifier(random_state=337)
model.fit(x_train_scaled, y_train)

# 6. 예측 및 정확도 평가
y_predict = model.predict(x_test_scaled)
acc = accuracy_score(y_test, y_predict)
print(f"======== XGBClassifier ==")
print("Original accuracy_score: ", acc)

# 7. 피처 임포턴스 추출 및 하위 20~25% 컬럼 제거
feature_importances = model.feature_importances_
threshold = np.percentile(feature_importances, 25)  # 하위 25% 임계값

# 중요도가 낮은 컬럼을 필터링
low_importance_features = [feature for feature, importance in zip(df.columns, feature_importances) if importance <= threshold]

# 컬럼 제거 후 데이터셋 재구성
df_reduced = df.drop(columns=low_importance_features)

# 8. 데이터 분할 및 스케일링 (재구성된 데이터셋)
x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    df_reduced, y, train_size=0.8, shuffle=True, random_state=337, stratify=y
)
x_train_reduced_scaled = scaler.fit_transform(x_train_reduced)
x_test_reduced_scaled = scaler.transform(x_test_reduced)

# 9. 재구성된 데이터셋으로 모델 재훈련 및 평가
model.fit(x_train_reduced_scaled, y_train_reduced)
y_predict_reduced = model.predict(x_test_reduced_scaled)
acc_reduced = accuracy_score(y_test_reduced, y_predict_reduced)

# 10. 결과 비교
print(f"Reduced accuracy_score: {acc_reduced}")
print(f"Removed features: {low_importance_features}")

# ======== XGBClassifier ==
# Original accuracy_score:  0.9333333333333333
# Reduced accuracy_score: 0.9333333333333333
# Removed features: ['sepal width (cm)']  