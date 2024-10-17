# m23_feature_importances01.py 복사

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (150, 4) (150,)

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

plot_feature_importances_dataset(model)
plt.show()

# random_state :  1223
# =================== DecisionTreeClassifier =====================
# acc 1.0
# [0.01666667 0.         0.57742557 0.40590776]
# =================== RandomForestClassifier =====================
# acc 1.0
# [0.10691492 0.02814393 0.42049394 0.44444721]
# =================== GradientBoostingClassifier =====================
# acc 1.0
# [0.01077099 0.01084882 0.27839833 0.69998185]
# =================== XGBClassifier =====================
# acc 1.0
# [0.00897023 0.02282782 0.6855639  0.28263798]