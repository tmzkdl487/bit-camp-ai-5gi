import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]]).astype('float64')

# print(data)
data = data.transpose()
# print(data)
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

print("============================================================")

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

imputer = IterativeImputer()    # 디폴트 BayesianRidge 회귀모델.
data1 = imputer.fit_transform(data)
print(data1)
# [[ 2.          2.          2.          2.0000005 ] 
#  [ 4.00000099  4.          4.          4.        ] 
#  [ 6.          5.99999928  6.          5.9999996 ] 
#  [ 8.          8.          8.          8.        ] 
#  [10.          9.99999872 10.          9.99999874]]

print("============================================================")

imputer = IterativeImputer(estimator = DecisionTreeRegressor())    
data2 = imputer.fit_transform(data)
print(data2)
# [[ 2.  2.  2.  4.]
#  [ 2.  4.  4.  4.]
#  [ 6.  8.  6.  8.]
#  [ 8.  8.  8.  8.]
#  [10.  8. 10.  8.]]

print("============================================================")

imputer = IterativeImputer(estimator = RandomForestRegressor())    
data3 = imputer.fit_transform(data)
print(data3)
# [[ 2.    2.    2.    4.96]
# [ 4.22  4.    4.    4.  ]
# [ 6.    4.54  6.    4.96]
# [ 8.    8.    8.    8.  ]
# [10.    6.92 10.    7.24]]

print("============================================================")

imputer = IterativeImputer(estimator = XGBRegressor())    
data4 = imputer.fit_transform(data)
print(data4)
# [[ 2.          2.          2.          4.00096321]
# [ 2.00112057  4.          4.          4.        ] 
# [ 6.          4.00000906  6.          4.00096321]
# [ 8.          8.          8.          8.        ]
# [10.          7.99906492 10.          7.99903679]]

