# 02_california
# 03_diabetse

# 06_cancer
# 09_wine
# 11_digits

from sklearn.datasets import fetch_california_housing

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRFRegressor

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (150, 4) (150,)

random_state = 1223

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
    print('r2', model.score(x_test, y_test))
    print(model.feature_importances_)


# print(model)

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

# random_state :  1223
# =================== DecisionTreeRegressor =====================
# r2 0.5964140465722068
# [0.51873533 0.05014494 0.05060456 0.02551158 0.02781676 0.13387334
#  0.09833673 0.09497676]
# =================== RandomForestRegressor =====================
# r2 0.811439104037621
# [0.52445075 0.05007899 0.04596161 0.03031591 0.03121773 0.1362301
#  0.09138102 0.09036389]
# =================== GradientBoostingRegressor =====================
# r2 0.7865333436969877
# [0.60051609 0.02978481 0.02084099 0.00454408 0.0027597  0.12535772
#  0.08997582 0.12622079]
# =================== XGBRFRegressor =====================
# r2 0.6973284037291707
# [0.451661   0.05509751 0.19447608 0.04169037 0.01094069 0.14598809
#  0.05534564 0.04480066]