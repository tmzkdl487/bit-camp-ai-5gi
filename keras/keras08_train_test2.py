import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [실습]넘파이 리스트의 슬라이싱!!! 7:3으로 잘라라!!

# x_train = x[0:7]
# x_train = x[:7]
# x_train = x[:-3]
x_train = x[0:-3]
print(x_train)  #[1 2 3 4 5 6 7]

# x_test = x[7:10]
# x_test = x[7:]
x_test = x[-3:]
print(x_test)   #[ 8  9 10]
