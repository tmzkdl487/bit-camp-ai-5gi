from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target 

print(x)    
print(y)
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=8989)

#2. 모델 구성
model = Sequential()
model.add(Dense(251, input_dim=10))
model.add(Dense(141))
model.add(Dense(171))
model.add(Dense(14))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True,
)

hist = model.fit(x_train, y_train, epochs=10, batch_size=32, 
          verbose=0, validation_split=0.2,
          callbacks= [es]
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print ("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

print("====================== hist =========================")
print(hist)
print("=================== hist.histroy ====================")
print(hist.history)
print("======================= loss =======================")
print(hist.history['loss'])
print("====================== val_loss ======================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c = 'red', label = 'val_loss')
plt.plot(hist.history['val_loss'], c = 'blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('Diabetes')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

# [실습] 맹그러봐 / R2 0.62 이상
# r2스코어 :  0.5189042303718636
# r2스코어 :  0.5665473691683233
# r2스코어 :  0.6007596386970162
# r2스코어 :  0.6006874672342544
# r2스코어 :  0.6151010366297509
# r2스코어 :  0.6180262183555738
# r2스코어 :  0.6196877963446061
# r2스코어 :  0.6211853813430838