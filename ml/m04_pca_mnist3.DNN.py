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

# 스케일링. 이거 추가.
x = x/255. # minmas 스케일링함.

# print(np.min(x), np.max(x))    # 0.0 1.0

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

pca = PCA(n_components=784)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) # cumsum은 누적합

# print(cumsum)

# print(np.argmax(cumsum >= 0.95) +1)  # 154
# print(np.argmax(cumsum >= 0.99) +1)  # 331
# print(np.argmax(cumsum >= 0.999) +1) # 486
# print(np.argmax(cumsum >= 1.0) +1)   # 713


  
