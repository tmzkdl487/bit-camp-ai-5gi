from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (569, 30) (569,)

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
# acc 0.9473684210526315
# [0.         0.05030732 0.         0.         0.         0.
#  0.         0.         0.         0.0125215  0.         0.03023319
#  0.         0.         0.         0.         0.00785663 0.
#  0.         0.         0.72931244 0.         0.0222546  0.01862569
#  0.01611893 0.         0.         0.0955152  0.01725451 0.        ]
# =================== RandomForestClassifier =====================
# acc 0.9298245614035088
# [0.02086793 0.01067931 0.04449403 0.05946836 0.00319558 0.02670713
#  0.02937097 0.0699787  0.00244273 0.0020165  0.02515119 0.00295319
#  0.00331962 0.02345914 0.00409896 0.00361462 0.00490543 0.00386996
#  0.00322823 0.00348085 0.12409483 0.0202164  0.14903546 0.1282556
#  0.01500504 0.01531022 0.03516301 0.15135923 0.0075333  0.00672446]
# =================== GradientBoostingClassifier =====================
# acc 0.9385964912280702
# [1.94023511e-04 2.49452840e-02 7.39095658e-04 1.11882290e-03
#  0.00000000e+00 5.15800222e-06 1.45512274e-02 4.88880805e-02
#  2.04936871e-05 2.09002025e-03 5.80367261e-04 8.49565095e-03
#  2.76122363e-03 1.54209684e-03 1.23141673e-03 0.00000000e+00
#  1.15082117e-05 1.07917966e-04 7.78817409e-05 2.26872696e-03
#  5.44493032e-01 3.78660080e-02 1.65403261e-01 2.52000516e-02
#  5.89157185e-03 4.95288035e-05 5.15995307e-03 1.05303238e-01
#  9.20938168e-04 8.34215822e-05]
# =================== XGBClassifier =====================
# acc 0.9385964912280702
# [0.01410364 0.01792158 0.         0.         0.         0.01414043
#  0.00264063 0.05220545 0.00094232 0.00426078 0.00051078 0.01685394
#  0.         0.01108845 0.00405639 0.00049153 0.00128015 0.00403847
#  0.00186279 0.00101614 0.38543397 0.01304734 0.3068675  0.01161618
#  0.01369678 0.         0.01367813 0.08019859 0.0109833  0.01706486]