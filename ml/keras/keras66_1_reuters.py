from tensorflow.keras.datasets import reuters

import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,
    # maxlen=10,          # <- 자르는 갯수
    test_split=0.2,
)

# print(x_train)
# [list([1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 2, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 2, 2, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 2, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12])
# 리스트의 형태이다.

# print(x_train.shape, x_test.shape)  # (8982,) (2246,)
# print(y_train.shape, y_test.shape)  # (8982,) (2246,)

# print(y_train)  # [ 3  4  3 ... 25  3 25]
# print(len(np.unique(y_train)))
# print(np.unique(y_train)) 하면 아래와 같이 나옴.
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# (len(np.unique(y_train))) len 치니 46 이라고 나옴.

# print(type(x_train))    # <class 'numpy.ndarray'> x_train의 타입확인
# print(type(x_train[0])) # <class 'list'> x_train의 0번째 타입 찾아봄.

# print(len(x_train[0]), len(x_train[1])) # 87 56

# print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train))  # 뉴스기사의 최대길이 :  2376
# print("뉴스기사의 최소길이 : ", min(len(i) for i in x_train))  # 뉴스기사의 최소길이 :  13
# print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train))    
# np.mean(len(i) for i in x_train))하면 TypeError: unsupported operand type(s) for /: 'generator' and 'int'에러씀
# sum(map(len, x_train)) / len(x_train))라는 함수를 쓰면 바로 뉴스기사의 평균길이 :  145.5398574927633 가 나옴. <- 함수 점심 시간에 찾아봄.

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, 
                        truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, 
                        truncating='pre')

# print(x_train.shape, x_test.shape)  # (8982, 100) (2246, 100)

# y 원핫하고 맹그러봐!!!!
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape)    # (8982, 46) (2246, 46)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(1000, 10))
# model.add(Bidirectional(LSTM(128)))  # return_sequences=True    
model.add(LSTM(64, activation='relu'))
model.add(Dense(46, activation='relu'))  
# model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))    # activation='relu'
model.add(Dense(46))
model.add(Dense(46, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights= True
)

model.fit(x_train, y_train, epochs=100, batch_size=100, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("ACC : ", results)

# ACC :  [1.7155715227127075, 0.638913631439209]  <- [0]로스 / [1] ACC
# ACC :  [1.7558889389038086, 0.6380231380462646] <- 2트
# ACC :  [2.1687519550323486, 0.6620659232139587] <- batch_size=32 함.
# ACC :  [1.8115798234939575, 0.6242208480834961] <- batch_size=200
# ACC :  [1.7271109819412231, 0.6162065863609314] <- batch_size=300
# ACC :  [1.6397143602371216, 0.6211041808128357] <- Bidirectional에 LSTM 넣음
# ACC :  [1.5985162258148193, 0.6798753142356873] <- 모델 64
# ACC :  [1.4409241676330566, 0.6691896915435791] <- 모델 128
# ACC :  [1.4268157482147217, 0.6736420392990112] <- Bidirectional 뺌.
# ACC :  [1.3712891340255737, 0.6794301271438599] <- model.add(Dropout(0.1)) 뺌.
# ACC :  [1.3704913854599, 0.6874443292617798]    <- 노드도 갯수 뺌
# ACC :  [1.8307873010635376, 0.687889575958252]  <- batch_size=100, activation='relu'뺌
# ACC :  [1.820429801940918, 0.6847729086875916]  <- batch_size=100, activation='relu'뺌 2트
# ACC :  [1.8411552906036377, 0.6754229664802551] <- activation='relu'넣어서 다시 돌림
# ACC :  [2.030789613723755, 0.6869990825653076]  <- activation='relu'뺀 노드 1개 추가
# ACC :  [1.457714319229126, 0.6821014881134033]  <- 코랩 점수
# ACC :  [2.231283664703369, 0.6794301271438599]  <- activation='relu' 3개 넣음.
# 