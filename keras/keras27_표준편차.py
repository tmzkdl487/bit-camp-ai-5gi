import numpy as np
from sklearn.preprocessing import StandardScaler

#1. 데이터
data = np.array([[1,2,3,1], 
                 [4,5,6,2],
                 [7,8,9,3],
                 [10,11,12,114],
                 [13,14,15,115]])    # 평균: 7, 8, 9, 47

#1. 평균
means = np.mean(data, axis=0)
print('평균 : ', means) # 평균 :  [ 7.  8.  9. 47.]

#2. 모집단 분산 (n빵 -> n은 한개의 열의 데이터의 갯수 예를 들면 이 데이터의 n은 5개.)
population_variancse = np.var(data, axis=0)
print("모집단 분산 : ", population_variancse)  # 분산 :  [  18.   18.   18. 3038.]

#3. 표본 분산 (n-1 빵 -> n은 한개의 열의 데이터의 갯수 예를 들면 이 데이터의 n은 5. n-1이니 4)
variances = np.var(data, axis=0, ddof=1)
print("표본 분산 : ", variances)    # 표본 분산 :  [  22.5   22.5   22.5 3797.5]

#4. 표본 표준 편차 (표준분산에 루트를 씌운 값.)
std = np.std(data, axis=0, ddof=1)  
print("표준편자 : ", std)   # 표준편자 :  [ 4.74341649  4.74341649  4.74341649 61.62385902]

#5. StandardScalar 
scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("StandarScaler : \n", scaled_data)

# StandarScaler :
#  [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]