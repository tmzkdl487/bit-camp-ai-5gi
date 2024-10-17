##############################
## 성능 좋은 것을 기준으로 해당 컬럼을 PCA로 만든 후 합치기

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#1. data
datasets= load_digits()

x = datasets.data
y = datasets.target
# y = pd.DataFrame(data=datasets.target)

from sklearn.model_selection import train_test_split
random_state=1223

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random_state)

#2. model
model = RandomForestClassifier(random_state=random_state)

model.fit(x_train, y_train)
print("===================", model.__class__.__name__, "====================")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

len = len(datasets.feature_names)
cut = round(len*0.2) ## 하위 20% 컬럼 갯수
print("하위 20% 컬럼 갯수 :", cut)
print(datasets.feature_names)
print('컬럼 제거 전 acc :', model.score(x_test, y_test))

percent = np.percentile(model.feature_importances_, 20)

rm_index=[]

for index, importance in enumerate(model.feature_importances_):
    if importance<=percent :
        rm_index.append(index)

#구린거
x_train1 = []     
for i in rm_index : 
    x_train1.append(x_train[:,i])
x_train1 = np.array(x_train1).T

x_test1 = []     
for i in rm_index : 
    x_test1.append(x_test[:,i])
x_test1 = np.array(x_test1).T

# 구린거 삭제한거
x_train = np.delete(x_train, rm_index, axis=1)
x_test = np.delete(x_test, rm_index, axis=1)

#구린거 1개로 합체
pca = PCA(n_components = 1)
x_train1 = pca.fit_transform(x_train1)
x_test1 = pca.transform(x_test1) 

#구린거+구린거 삭제한거
x_train = np.concatenate((x_train,x_train1),axis=1)
x_test = np.concatenate((x_test, x_test1),axis=1)


model = RandomForestClassifier(random_state=random_state)

model.fit(x_train, y_train)
print('PCA하고 합친 acc :', model.score(x_test, y_test))

# =================== RandomForestClassifier ====================
# acc : 0.9611111111111111
# [0.00000000e+00 2.27255374e-03 2.39409828e-02 9.74782341e-03
#  9.82864541e-03 2.15818480e-02 9.38757634e-03 9.04124415e-04
#  3.44737020e-05 1.13604400e-02 2.26963485e-02 7.30694430e-03
#  1.22274358e-02 3.20217428e-02 4.90199097e-03 6.52085082e-04
#  2.59927619e-05 8.57233696e-03 2.09972208e-02 2.74905815e-02
#  2.85842804e-02 5.18454176e-02 1.00426676e-02 4.01521936e-04
#  3.88537353e-05 1.39585435e-02 4.27014098e-02 2.70785396e-02
#  2.95212069e-02 2.27825980e-02 3.02504257e-02 0.00000000e+00
#  0.00000000e+00 2.85987443e-02 2.66123542e-02 1.88222442e-02
#  4.13541302e-02 1.64535185e-02 2.48991464e-02 0.00000000e+00
#  1.10801840e-04 1.18413235e-02 3.80324750e-02 4.51863797e-02
#  1.86937781e-02 1.96939666e-02 1.98331096e-02 2.13981323e-05
#  6.87543264e-06 2.14131393e-03 1.78026945e-02 1.84267237e-02
#  1.47624491e-02 2.04667589e-02 2.66593345e-02 1.58749974e-03
#  2.29105883e-05 1.44583541e-03 2.00160596e-02 1.01661179e-02
#  2.07438790e-02 3.15202026e-02 1.84719635e-02 2.44939916e-03]
# 하위 20% 컬럼 갯수 : 13
# ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7', 'pixel_2_0', 
# 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7']
# 컬럼 제거 전 acc : 0.9611111111111111
# PCA하고 합친 acc : 0.9666666666666667
