from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# 1. 데이터셋 로드
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# 2. 데이터 프레임으로 변환
df = pd.DataFrame(x, columns=cancer.feature_names)

# 3. 데이터 분할
random_state = 1223
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random_state)

# 4. 모델 정의 및 훈련
model = RandomForestClassifier(random_state=random_state)
model.fit(x_train, y_train)

# 5. 모델 평가 및 중요도 분석
print("===================", model.__class__.__name__, "====================")
print('Original accuracy_score:', model.score(x_test, y_test))
print('Feature Importances:', model.feature_importances_)

# 하위 20% 중요도 피처 제거
num_features = x.shape[1]
cut = round(num_features * 0.2)  # 하위 20% 컬럼 갯수
percentile_threshold = np.percentile(model.feature_importances_, 20)

# 중요도가 낮은 피처의 인덱스 찾기
rm_index = [index for index, importance in enumerate(model.feature_importances_) if importance <= percentile_threshold]

print("하위 20% 컬럼 갯수:", len(rm_index))
print('Low importance features indices:', rm_index)
print('Low importance feature names:', np.array(cancer.feature_names)[rm_index])

# 중요도가 낮은 피처를 따로 저장
x_train_low = x_train[:, rm_index]
x_test_low = x_test[:, rm_index]

# 중요도가 낮은 피처 제거 후 데이터셋 재구성
x_train_reduced = np.delete(x_train, rm_index, axis=1)
x_test_reduced = np.delete(x_test, rm_index, axis=1)

# PCA를 적용하여 중요도가 낮은 피처들을 하나로 통합
pca = PCA(n_components=1)
x_train_pca = pca.fit_transform(x_train_low)
x_test_pca = pca.transform(x_test_low)

# PCA의 설명력 확인
print(f"Explained variance ratio of PCA component: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

# PCA로 변환된 특성을 원래 데이터셋에 추가
x_train_combined = np.concatenate((x_train_reduced, x_train_pca), axis=1)
x_test_combined = np.concatenate((x_test_reduced, x_test_pca), axis=1)

# 모델 재훈련 및 평가
model = RandomForestClassifier(random_state=random_state)
model.fit(x_train_combined, y_train)
accuracy_after_pca = model.score(x_test_combined, y_test)

# 결과 비교
print('Accuracy after removing low importance features and applying PCA:', accuracy_after_pca)

# =================== RandomForestClassifier ====================
# Original accuracy_score: 0.9824561403508771
# Feature Importances: [0.01032336 0.01429674 0.0242272  0.06564872 0.00521536 0.02501443
#  0.04386496 0.09773562 0.00324672 0.00364132 0.02370795 0.00358133
#  0.00723111 0.0285759  0.00448963 0.00506061 0.00602691 0.00326417
#  0.00446753 0.00389223 0.08437281 0.01579862 0.14665637 0.12312715
#  0.01165522 0.01756141 0.03562563 0.166519   0.00901227 0.00615971]
# 하위 20% 컬럼 갯수: 6
# Low importance features indices: [8, 9, 11, 17, 18, 19]
# Low importance feature names: ['mean symmetry' 'mean fractal dimension' 'texture error'
#  'concave points error' 'symmetry error' 'fractal dimension error']
# Explained variance ratio of PCA component: [0.99706001]
# Total explained variance: 0.9970600069322749
# Accuracy after removing low importance features and applying PCA: 0.9824561403508771
