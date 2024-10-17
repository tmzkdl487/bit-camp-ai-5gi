from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (1797, 64) (1797,)

random_state = 1223

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    random_state=random_state, 
                                                    stratify=y,)

#2. 모델구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

print("random_state : ", random_state)
for model in models: 
    model.fit(x_train, y_train)
    # model_name = type(model).__name__ # 챗GPT
    # print("===================", model_name, "=====================")
    print("===================", model.__class__.__name__, "=====================") # 누리님 버전
    print('acc', model.score(x_test, y_test))
    print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

# print(model)

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)

# plot_feature_importances_dataset(model)
# plt.show()

# [실습] 그림 4개 한페이지에 넣어라!!! 맹그러!!!!!

plt.subplot(2, 2, 1)
plot_feature_importances_dataset(model1)

plt.subplot(2, 2, 2)
plot_feature_importances_dataset(model2)

plt.subplot(2, 2, 3)
plot_feature_importances_dataset(model3)

plt.subplot(2, 2, 4)
plot_feature_importances_dataset(model4)

plt.show()