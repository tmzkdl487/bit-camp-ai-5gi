import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

# print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data)
#     x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 0. 결측치 확인
# print(data.isnull())          # 데이터프레임 내에서 결측값이 있는지 여부를 True/False로 표시
#      x1     x2     x3     x4
# 0  False  False  False   True
# 1   True  False  False  False
# 2  False   True  False   True
# 3  False  False  False  False
# 4  False   True  False   True

# print(data.isnull().sum())    # 결과: 각 열마다 결측값이 몇 개 있는지 합계
# x1    1
# x2    2
# x3    0
# x4    3
# dtype: int64

# print(data.info())    #  데이터프레임의 구조, 각 열의 데이터 타입, 결측값의 개수 등을 간단하게 보여줌
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
# 0   x1      4 non-null      float64
# 1   x2      3 non-null      float64
# 2   x3      5 non-null      float64
# 3   x4      2 non-null      float64
# dtypes: float64(4)
# memory usage: 288.0 bytes
# None

# 1. 결측치 삭제
# print(data.dropna())      # 디폴트는 axis = 0
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0

# print(data.dropna(axis=0)) # Ture 있는 행삭제
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0

# print(data.dropna(axis=1)) # Ture 있는 열 삭제
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

# 2-1. 특정값 - 평균
# means = data.mean()
# print(means)
# x1    6.500000
# x2    4.666667
# x3    6.000000
# x4    6.000000
# dtype: float64

# data2 = data.fillna(means)
# print(data2)
#      x1        x2    x3   x4
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0

# 2-2. 특정값 - 중위값
# med = data.median()
# print(med)
# x1    7.0
# x2    4.0
# x3    6.0
# x4    6.0
# dtype: float64

# data3 = data.fillna(med)
# print(data3)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0
    
# 2-3. 특정값 - 0 채우기 / 임의의 값 채우기
# data4 = data.fillna(0)
# print(data4)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0

# data4_2 = data.fillna(777)
# print(data4_2)
#       x1     x2    x3     x4
# 0    2.0    2.0   2.0  777.0
# 1  777.0    4.0   4.0    4.0
# 2    6.0  777.0   6.0  777.0
# 3    8.0    8.0   8.0    8.0
# 4   10.0  777.0  10.0  777.0

# 2-4. 특정값 -ffill <- forward fill, 앞쪽 값으로 채우기 / 이전 값(앞쪽에 있는 값)으로 결측값을 채움.
# data5 = data.ffill()
# data5 = data.fillna(method='ffill') # 위에랑 같은 말임.
# print(data5)
#      x1   x2    x3   x4
#0   2.0  2.0   2.0  NaN
#1   2.0  4.0   4.0  4.0
#2   6.0  4.0   6.0  4.0
#3   8.0  8.0   8.0  8.0
#4  10.0  8.0  10.0  8.0

# 2-5. 특정값 - bfill <- backward fill, 뒤쪽 값으로 채우기
# data6 = data.bfill()
# data6 = data.fillna(method='bfill') # 위랑 같은 코드임. 이런식으로도 쓸 수 있음.
# print(data6)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

############ 특정 칼럼만 ####################################
# means = data['x1'].mean()
# print(means) # 6.5

# meds = data['x4'].median()
# print(meds)  # 6.0

# data['x1'] = data['x1'].fillna(means)
# data['x4'] = data['x4'].fillna(meds)
# data['x2'] = data['x2'].ffill()

# print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   6.5  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  6.0
