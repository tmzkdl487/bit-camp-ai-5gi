import pandas as pd
import numpy as np

path = 'C:/ai5/_data/kaggle/netflix/'
train_csv = pd.read_csv(path + 'train.csv')
print(train_csv)    # [967 rows x 6 columns]
print(train_csv.info()) 
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 967 entries, 0 to 966
# Data columns (total 6 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   Date    967 non-null    object
#  1   Open    967 non-null    int64
#  2   High    967 non-null    int64
#  3   Low     967 non-null    int64
#  4   Volume  967 non-null    int64
#  5   Close   967 non-null    int64
# dtypes: int64(5), object(1)
# memory usage: 45.5+ KB
# None

print(train_csv.describe())