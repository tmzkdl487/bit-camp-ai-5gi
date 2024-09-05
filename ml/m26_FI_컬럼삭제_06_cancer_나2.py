from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. 데이터 로드
datasets = load_breast_cancer()
df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
x = df.values
y = datasets.target

# 데이터 분할
random_state = 1223
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state)

# 2. 모델 구성 및 훈련
model = XGBClassifier(random_state=random_state)
model.fit(x_train, y_train)

# 3. 모델 평가 및 중요도 분석
feature_importances = model.feature_importances_
sorted_idx = np.argsort(feature_importances)

# 원본 모델 평가
original_accuracy = accuracy_score(y_test, model.predict(x_test))
print(f"Original Accuracy Score: {original_accuracy}")

# 하위 20% 및 25% 중요도 컬럼 제거 및 평가
def remove_low_importance_features(x_train, x_test, feature_importances, percentage):
    sorted_idx = np.argsort(feature_importances)
    n_remove = int(len(sorted_idx) * (percentage / 100))
    removed_features_idx = sorted_idx[:n_remove]
    
    x_train_reduced = np.delete(x_train, removed_features_idx, axis=1)
    x_test_reduced = np.delete(x_test, removed_features_idx, axis=1)
    
    return x_train_reduced, x_test_reduced, removed_features_idx

# 평가
for percentage in [20, 25]:
    x_train_reduced, x_test_reduced, removed_features_idx = remove_low_importance_features(x_train, x_test, feature_importances, percentage)
    
    # 모델 재훈련 및 평가
    model_reduced = XGBClassifier(random_state=random_state)
    model_reduced.fit(x_train_reduced, y_train)
    
    reduced_accuracy = accuracy_score(y_test, model_reduced.predict(x_test_reduced))
    
    print(f"\nAccuracy Score after removing {percentage}% lowest importance features: {reduced_accuracy}")

    # 출력 제거된 특성
    removed_features = np.array(datasets.feature_names)[removed_features_idx]
    print(f"Removed features: {removed_features}")

# Original Accuracy Score: 0.956140350877193

# Accuracy Score after removing 20% lowest importance features: 0.9824561403508771
# Removed features: ['mean perimeter' 'perimeter error' 'mean symmetry' 'concave points error'
#  'mean radius' 'symmetry error']

# Accuracy Score after removing 25% lowest importance features: 0.9824561403508771
# Removed features: ['mean perimeter' 'perimeter error' 'mean symmetry' 'concave points error'
#  'mean radius' 'symmetry error' 'texture error']