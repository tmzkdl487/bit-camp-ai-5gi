from sklearn.datasets import load_wine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# 1. 데이터셋 로드
wine = load_wine()
x, y = wine.data, wine.target

# 데이터프레임으로 변환
df = pd.DataFrame(x, columns=wine.feature_names)

# 2. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    df, y, train_size=0.8, shuffle=True, random_state=337, stratify=y
)

# 3. 데이터 스케일링
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 4. 모델 정의 및 훈련
model = XGBClassifier(random_state=337)
model.fit(x_train_scaled, y_train)

# 5. 예측 및 정확도 평가
y_predict = model.predict(x_test_scaled)
original_acc = accuracy_score(y_test, y_predict)
print("Original accuracy_score: ", original_acc)

# 6. 피처 임포턴스 추출 및 하위 중요도 컬럼 제거
feature_importances = model.feature_importances_
threshold = np.percentile(feature_importances, 40)  # 하위 40% 임계값

# 중요도가 낮은 컬럼을 필터링
low_importance_features = [feature for feature, importance in zip(df.columns, feature_importances) if importance <= threshold]
print(f"Low importance features: {low_importance_features}")

# 컬럼 제거 후 데이터셋 재구성
df_reduced = df.drop(columns=low_importance_features)
x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    df_reduced, y, train_size=0.8, shuffle=True, random_state=337, stratify=y
)
x_train_reduced_scaled = scaler.fit_transform(x_train_reduced)
x_test_reduced_scaled = scaler.transform(x_test_reduced)

# 7. PCA 적용
# 하위 중요도 피처만을 가지고 PCA 적용
x_train_removed = x_train_scaled[:, [df.columns.get_loc(feature) for feature in low_importance_features]]
x_test_removed = x_test_scaled[:, [df.columns.get_loc(feature) for feature in low_importance_features]]

# PCA에서 설명력을 충분히 확보할 수 있도록 n_components를 조정
pca = PCA(n_components=min(5, x_train_removed.shape[1]))  # 예를 들어, 5개 컴포넌트 사용
x_train_pca = pca.fit_transform(x_train_removed)
x_test_pca = pca.transform(x_test_removed)

# PCA의 설명력 확인
print(f"Explained variance ratio of PCA components: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

# PCA로 변환된 특성을 원래 데이터셋에 추가
x_train_combined = np.hstack((x_train_reduced_scaled, x_train_pca))
x_test_combined = np.hstack((x_test_reduced_scaled, x_test_pca))

# 8. 모델 재훈련 및 평가
model.fit(x_train_combined, y_train_reduced)
y_predict_combined = model.predict(x_test_combined)
reduced_acc = accuracy_score(y_test_reduced, y_predict_combined)

# 9. 결과 비교
print(f"Accuracy after removing low importance features and applying PCA: {reduced_acc}")

# Original accuracy_score:  1.0
# Low importance features: ['alcohol', 'alcalinity_of_ash', 'nonflavanoid_phenols', 'proanthocyanins', 'hue']    
# Explained variance ratio of PCA components: [0.41059872 0.23243714 0.16207991 0.11241948 0.08246475]
# Total explained variance: 1.0
# Accuracy after removing low importance features and applying PCA: 1.0