# keras26_scaler06_cancer.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = load_breast_cancer()
# print (datasets)

# print(datasets.DESCR)   # DESCR은 교육용만 쓴다.
# print(datasets.feature_names)   # 30개의 데이터가 있다.
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

x = datasets.data
y = datasets.target
# [검색] 넘파이 라벨값 찾는 법. (종류 구하는 법, 0과 1의 갯수가 몇개인지 찾아요.)
# [힌트] 넘파이는 유니크 / 판다스는 벨류 카운트

# print(x.shape, y.shape) # (569, 30) (569,) <- 넘파이 데이터
# print(type(x))  # <class 'numpy.ndarray'>라고 나옴.

# print(y.value_counts())  # 에러 남. 판다스에만 됨. 그래서 아래 3개로 알 수 있음.
# print(pd.DataFrame(y). value_counts())  # 판다스에서 카운트 세는 법. print(pd.Series(y). value_counts()) 랑 print(pd.value_counts(y))도 다 똑같다.
#  1    357
#  0    212

# print(pd.Series(y). value_counts()) 
# print(pd.value_counts(y))

# print(np.unique(y, return_counts=True)) 
# (array([0, 1]), array([212, 357], dtype=int64))
# 2진 분류할 때 갯수를 왜 구할까? 불균형 데이터인지 확인하려고.

# print(x.shape, y.shape) # (569, 30) (569,)

# exit()

pca = PCA(n_components=8)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 1
# print(np.argmax(cumsum >= 0.99) +1)  # 2
# print(np.argmax(cumsum >= 0.999) +1) # 3
# print(np.argmax(cumsum >= 1.0) +1)   # 1

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.9, 
                                                    random_state= 3434)


n = [1, 2, 3]

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
    model.add(Dense(1))   # activation='softmax'
    
    #3. 컴파일, 훈련
    model.compile(loss= 'mse', optimizer='adam', metrics=['acc'])

    start = time.time()
    
    es = EarlyStopping(monitor='val_loss', mode='min',
                    patience=5, verbose=0,
                    restore_best_weights=True)
    
    ########################### mcp 세이프 파일명 만들기 시작 ################
    import datetime 
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/m05/'
    filename = '{epoch:04d}-{val_loss:4f}.hdf5' # '1000-0.7777.hdf5'    
    filepath = "".join([path, 'm04_03_date_', str(i+1),'_', date, '_epo_', filename])
    ########################### mcp 세이프 파일명 만들기 끗 ################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    model.fit(x_train1, y_train, epochs=10, batch_size=64, verbose=0, validation_split=0.2,
              callbacks=[es, mcp])
    
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)
    
    print('===========================')
    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    
# ===========================
# 결과 PCA : 1
# acc :  0.8947368264198303
# 걸린 시간 :  1.45 초
# ===========================
# 결과 PCA : 2
# acc :  0.9298245906829834
# 걸린 시간 :  0.78 초
# ===========================
# 결과 PCA : 3
# acc :  0.9298245906829834
# 걸린 시간 :  0.77 초