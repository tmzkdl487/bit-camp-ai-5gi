from sklearn.datasets import load_diabetes

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRFRegressor

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (442, 10) (442,)

random_state = 1223

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    random_state=random_state, 
                                                    # stratify=y,
                                                    )

#2. 모델구성
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRFRegressor(random_state=random_state)

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

# =================== DecisionTreeRegressor =====================
# acc -0.24733855513252667
# [0.05676749 0.01855931 0.23978058 0.08279462 0.05873671 0.0639961
#  0.04130515 0.01340568 0.33217096 0.0924834 ]
# =================== RandomForestRegressor =====================
# acc 0.3687286985683689
# [0.05394197 0.00931513 0.25953258 0.1125408  0.04297661 0.05293764
#  0.06684433 0.02490964 0.29157054 0.08543076]
# =================== GradientBoostingRegressor =====================
# acc 0.3647974813076822
# [0.04509096 0.00780692 0.25858035 0.09953666 0.02605597 0.06202725
#  0.05303144 0.01840481 0.35346141 0.07600423]
# =================== XGBRFRegressor =====================
# acc 0.33589682441838786
# [0.02255717 0.02129483 0.1675215  0.06622679 0.04581491 0.05392584
#  0.05983198 0.07966285 0.39199358 0.09117053]