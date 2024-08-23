# keras68_optimizer01_boston.py 카피

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau    # Plateau 고원, 높고 편평한 땅

es = EarlyStopping (monitor='val_loss', mode='min',
                    patience=30, verbose=1,
                    restore_best_weights=True,)

rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                        patience=15, verbose=1, 
                        factor=0.9)

from tensorflow.keras.optimizers import Adam    
# 로스를 최적화하기 위해서 아담을 쓴다.

learning_rate = 0.05   # 디폴트: 0.001

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1000,
          batch_size=32, 
          callbacks=[es, rlr],
          )

#4. 평가, 예측
print("======================= 1. 기본출력 =============================")

loss = model.evaluate(x_test, y_test, verbose=0)
print('lr: {0}, 로스:{0}'.format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0)

r2 = r2_score(y_test, y_predict)
print('lr: {0}, r2: {1}'.format(learning_rate, r2))

# ======================= 1. 기본출력 =============================
# lr: 0.01, 로스:0.01
# lr: 0.01, r2: 0.6298726307176181 <- learning_rate = 0.01

#======================= 1. 기본출력 =============================
# lr: 0.01, 로스:0.01
# lr: 0.01, r2: 0.6325753505022675

# ======================= 1. 기본출력 =============================
# lr: 0.01, 로스:0.01
# lr: 0.01, r2: 0.6325754038579643

# ======================= 1. 기본출력 =============================
# lr: 0.01, 로스:0.01
# lr: 0.01, r2: 0.6325753505022675

# ======================= 1. 기본출력 =============================
# lr: 0.05, 로스:0.05
# lr: 0.05, r2: 0.6097316567886202