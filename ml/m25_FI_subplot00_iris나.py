# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier

# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import numpy as np

# #1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets.target

# # print(x.shape, y.shape) # (150, 4) (150,)

# random_state = 1223

# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
#                                                     random_state=random_state, 
#                                                     stratify=y,)

# #2. 모델구성
# model1 = DecisionTreeClassifier(random_state=random_state)
# model2 = RandomForestClassifier(random_state=random_state)
# model3 = GradientBoostingClassifier(random_state=random_state)
# model4 = XGBClassifier(random_state=random_state)

# models = [model1, model2, model3, model4]
# model_names = [model.__class__.__name__ for model in models]

# # 모델 학습
# for model in models:
#     model.fit(x_train, y_train)

# # 시각화
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle('Feature Importances of Different Models', fontsize=16)

# for ax, model, name in zip(axes.ravel(), models, model_names):
#     n_features = datasets.data.shape[1]
#     feature_importances = model.feature_importances_
#     ax.barh(np.arange(n_features), feature_importances, align='center')
#     ax.set_yticks(np.arange(n_features))
#     ax.set_yticklabels(datasets.feature_names)
#     ax.set_xlabel('Feature Importances')
#     ax.set_title(name)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot to fit suptitle
# plt.show()

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

random_state = 1223

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    random_state=random_state, 
                                                    stratify=y,)

# 2. 모델 구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]
model_names = [model.__class__.__name__ for model in models]

# 모델 학습
for model in models:
    model.fit(x_train, y_train)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Importances of Different Models', fontsize=16)

for ax, model, name in zip(axes.ravel(), models, model_names):
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        feature_importances = np.zeros_like(datasets.data[0])
        
    n_features = datasets.data.shape[1]
    ax.barh(np.arange(n_features), feature_importances, align='center')
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(datasets.feature_names)
    ax.set_xlabel('Feature Importances')
    ax.set_title(name)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot to fit suptitle
plt.show()