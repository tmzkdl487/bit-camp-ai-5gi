from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (178, 13) (178,)

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

# random_state :  1223
# =================== DecisionTreeClassifier =====================
# acc 0.8611111111111112
# [0.02100275 0.         0.         0.03810533 0.         0.
#  0.13964046 0.         0.         0.         0.03671069 0.3624326
#  0.40210817]
# =================== RandomForestClassifier =====================
# acc 0.9444444444444444
# [0.13789135 0.02251876 0.01336314 0.03826336 0.02830375 0.05255915
#  0.14261827 0.00916645 0.03234439 0.13563367 0.07199803 0.13963923
#  0.17570046]
# =================== GradientBoostingClassifier =====================
# acc 0.9166666666666666
# [1.43484205e-01 4.06686613e-02 6.69215300e-03 1.66979468e-03
#  1.32685947e-02 1.46219235e-08 1.13207893e-01 9.28570988e-04
#  1.26289568e-03 1.61668502e-01 3.37186677e-03 2.48779007e-01
#  2.64997841e-01]
# =================== XGBClassifier =====================
# acc 0.9444444444444444
# [0.07716953 0.03067267 0.04416747 0.00285905 0.01373686 0.0016962
#  0.07846211 0.00365221 0.02516203 0.08851561 0.00581782 0.528609
#  0.09947944]