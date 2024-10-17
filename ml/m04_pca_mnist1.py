from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape)     # (60000, 28, 28) (10000, 28, 28)    28*28= 784
# print(y_train.shape, y_test.shape)     # (60000,) (10000,)

(x_train, _), (x_test, _) = mnist.load_data()   # x만 쓸껀데 자리는 맞춰주려고 _를 넣는 것임.
# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis=0)
# print(x.shape)  # (70000, 28, 28)

################################## [실습] ###############################
# pca를 통해 0.95 이상인 n_components는 몇 개?
# 0.95 이상
# 0.99 이상
# 0.999 이상
# 1.0 일 때 몇 개?

# 힌트 np.argmax
########################################################################
# x = x.reshape(70000, 28*28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print(x.shape)  # (70000, 784)

# exit()  

pca = PCA(n_components=600)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) # cumsum은 누적합

# print(cumsum)

# exit()

############ 챗GPT 버전 ############

# n_components_95 = np.argmax(cumsum >= 0.95) + 1 # argmax는 맥스값의 위치
# n_components_99 = np.argmax(cumsum >= 0.99) + 1
# n_components_999 = np.argmax(cumsum >= 0.999) + 1
# n_components_1 = np.argmax(cumsum >= 1.0) + 1

# print(f"n_components >= 0.95: {n_components_95}")   # n_components >= 0.95: 154
# print(f"n_components >= 0.99: {n_components_99}")   # n_components >= 0.99: 331
# print(f"n_components >= 0.999: {n_components_999}") # n_components >= 0.999: 486
# print(f"n_components >= 1.0: {n_components_1}")     # n_components >= 1.0: 713

############ 누리님 버전 ############

# print('0.95 이상 : ' , np.min(np.where(cumsum>=0.95)) + 1)  # 0.95 이상 :  153
# print('0.99 이상 : ', np.min(np.where(cumsum>=0.99)) + 1)  # 0.99 이상 :  330
# print('0.999 이상 :', np.min(np.where(cumsum>=0.999)) + 1) # 0.999 이상:  485
# print('1.0 일 때 :', np.argmax(cumsum))                     # 1.0 일 때 :  712
  
############ 선생님 버전 ############

print(np.argmax(cumsum >= 0.95) +1)  # 154
print(np.argmax(cumsum >= 0.99) +1)  # 331
print(np.argmax(cumsum >= 0.999) +1) # 486
print(np.argmax(cumsum >= 1.0) +1)   # 713
  
# import matplotlib.pyplot as plt
# plt.plot(evr_cumsum)
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.grid()
# plt.show()