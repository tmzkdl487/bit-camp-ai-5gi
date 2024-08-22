from tensorflow.keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000,
)

# print(x_train)
# [list([1, 14, 22, 16, 43, 530, 973, 2, 2, 65, 458, 2, 66, 2, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 2, 2, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2, 19, 14, 22, 4, 2, 2, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 2, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2, 2, 16, 480, 66, 2, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 
# 4, 2, 7, 2, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 2, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2, 56, 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 
# 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 2, 88, 12, 16, 283, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32])
# 리스트의 형태이다.

# print(x_train.shape, x_test.shape)  # (25000,) (25000,)
# print(y_train.shape, y_test.shape)  # (25000,) (25000,)

# print(y_train)  # [1 0 0 ... 0 1 0]
# print(len(np.unique(y_train)))  # 2
# print(np.unique(y_train)) 하면 아래와 같이 나옴.
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# (len(np.unique(y_train))) len 치니 46 이라고 나옴.

# print(type(x_train))    # <class 'numpy.ndarray'>
# print(type(x_train[0])) # <class 'list'>

# print(len(x_train[0]), len(x_train[1])) # 218 189

# print("imdb의 최대길이 : ", max(len(i) for i in x_train))  # imdb의 최대길이 :  2494
# print("imdb의 최소길이 : ", min(len(i) for i in x_train))  # imdb의 최소길이 :  11
# print("imdb의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # imdb의 평균길이 :  238.71364

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, 
                        truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, 
                        truncating='pre')

# print(x_train.shape, x_test.shape) # (25000, 100) (25000, 100)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape)  # (25000, 2) (25000, 2)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(1000, 10))
model.add(Bidirectional(LSTM(100, return_sequences=True)))  # return_sequences=True    
model.add(LSTM(50, activation='relu'))
model.add(Dense(46, activation='relu'))  
model.add(Dense(30, activation='elu'))    # activation='relu' , activation='elu'
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 5,
    restore_best_weights= True
)

model.fit(x_train, y_train, epochs=100, batch_size=1000, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("ACC : ", results)

# ACC :  [0.691031277179718, 0.5212799906730652]  <- MinMaxScaler, elu
# ACC :  [0.691318929195404, 0.5238400101661682]  <- StandardScaler
# ACC :  [0.6919286847114563, 0.5205199718475342] <- 모델 바꿈
# ACC :  [0.6915156841278076, 0.520359992980957]  <- 노드2개
# ACC :  [0.6916282176971436, 0.5192400217056274] <- 코랩
# ACC :  [0.6915256977081299, 0.5180799961090088] <- epochs=100, batch_size=1000
# ACC :  [0.6910677552223206, 0.5157999992370605] <- Bidirectional 
# ACC :  [0.6912825703620911, 0.5198000073432922] <- 