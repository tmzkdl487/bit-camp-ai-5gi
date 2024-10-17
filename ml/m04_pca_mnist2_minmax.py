# m04_1에서 뽑은 4가지 결과로
# 4가지 모델을 맹그러
# input_shape = ()
# 1. 70000, 154
# 2. 70000, 331
# 3. 70000, 486
# 4. 70000, 713
# 5. 70000, 784 원본

# 시간과 성능을 체크한다.

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from keras.layers import Dense

from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

x_train = x_train/255.
x_test = x_test/255.

# 데이터를 2D로 변환
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

# print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

n = [154, 331, 486, 713]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 결과 저장
results = []

for i in range(0, len(n), 1):
    pca = PCA(n_components=n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=n[i]))   # relu는 음수는 무조껀 0으로 만들어 준다.
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    #3. 컴파일, 훈련
    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

    start = time.time()
    
    es = EarlyStopping(monitor='val_loss', mode='min',
                    patience=5, verbose=1,
                    restore_best_weights=True)

    model.fit(x_train1, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.2,
              callbacks=[es])
    
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)
    
    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")

# 결과 예시 #############################
# 결과1. PCA=154
# 걸린시간 초
# acc = 

# 결과2. PCA=331
# 걸린시간 초
# acc = 

# 결과3. PCA=486
# 걸린시간 초
# acc = 

# 결과4. PCA=713
# 걸린시간 초
# acc = 

# 결과5. PCA=784
# 걸린시간 초
# acc = 

  
