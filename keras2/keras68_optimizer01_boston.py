from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

import random as rn
import tensorflow as tf
tf.random.set_seed(337)
np.random.seed(337)
rn.seed(337)

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, 
                                                     shuffle=True, 
                                                     random_state=337)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam    
# 로스를 최적화하기 위해서 아담을 쓴다.

# learning_rate = 0.01
# learning_rate = 0.005
# learning_rate = 0.001 # 디폴트.
# learning_rate = 0.0015
# learning_rate = 1.0
# learning_rate = 0.0007  # r2: 0.637158517418638
# learning_rate = 0.0008  # r2: 0.6373047472901204
# learning_rate = 0.0009  # r2: 0.6372752876248525
# learning_rate = 0.002
# learning_rate = 0.005
learning_rate = 0.0001

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=100,
          batch_size=32, 
          )

#4. 평가, 예측
print("======================= 1. 기본출력 =============================")

loss = model.evaluate(x_test, y_test, verbose=0)
print('lr: {0}, 로스:{0}'.format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0)

r2 = r2_score(y_test, y_predict)
print('lr: {0}, r2: {1}'.format(learning_rate, r2))

# 로스:34.0585289?00146484 / r2: 0.6372742798623421
# 로스:34.05852508544922 / r2: 0.6372742970185294

######################## [실습] ##############################
# lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

############# 다 맹그러 #################
# lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

#### 맹그러봐 아래꺼 10개 #######


# 1. m05_pca_evr_실습10_fetch_covtype.py

# 2. m05_pca_evr_실습12_kaaggle_santander.py
# 3. m05_pca_evr_실습13_kaggle_otto.py

# 4. m05_pca_evr_실습14_mnist.py
# 5. m05_pca_evr_실습15_fachion.py
# 6. m05_pca_evr_실습16_cifar10.py
# 7. m05_pca_evr_실습17_cifar100.py
# 8. m05_pca_evr_실습18_cat_dog.py 
# 8. m05_pca_evr_실습19_horse.py

# 10. m05_pca_evr_실습21_jena.py