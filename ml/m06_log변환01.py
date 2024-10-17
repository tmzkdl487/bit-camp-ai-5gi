import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=1000)

# print(data) 
# 1.91761270e+00 1.23243254e+00 9.34646132e-01 1.88154651e-01
#  1.60103333e+00 5.78191222e-01 4.42350688e+00 2.09747854e+00
#  1.37732495e+00 3.82181957e+00 9.80019973e-01 1.17811280e+00

# print(data.shape)   # (1000,)
# print(np.min(data), np.max(data))
# 0.0027604247286905433 16.219378126258395 <- 

# log_data = np.log(data) # <- 로그 0은 값이 없으니까.

log_data = np.log1p(data) # log1p를 넣어서 값이 나온 모든 숫자에 1을 더해서 만들어줘야한다.

# 원본 데이터 하스토그램 그리자
plt.subplot(1, 2, 1)     # 1행 2열 중 첫 번째 위치
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('original')
# plt.show()

# 로그변환 데이터 히스토그램 그리자
plt.subplot(1, 2, 2)    # 1행 2열 중 두 번째 위치
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed')

plt.show()
