# keras65_Embedding04_ohe_LSTM.py.py 복사

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, LSTM 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical

import numpy as np

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', ' 한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밋네요.',
    '준영이 바보', '반장 잘생겼다.', '태운이 또 구라친다.',
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])  # 1이 긍정, 0이 부정

token = Tokenizer()
token.fit_on_texts(docs)

x = token.texts_to_sequences(docs)

pad_x = pad_sequences(x, maxlen=5)

# print(pad_x.shape) # (15, 5)

# x_train = to_categorical(pad_x, num_classes=31)
# print(x_train)
# print(x_train.shape)    # (15, 5, 31)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding

# model = Sequential()
# model.add(Embedding(input_dim=31, output_dim=200, input_length=5))  

# input_dim 단어 사전의 갯수
# output_dim=100은 임의로 다음 노드에 넣어주니 바꿀 수 있음.
# input_length=은 5 바꿀 수 없음. 15, 5이라서 행무시 열 우선으로 5라서 

# model.summary() / output_dim=100
# Model: "sequential" 임배딩은 2차원으로 받아서 3차원으로 나옴./ 자연어처리, 시계열 데이터에서 많이씀.
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100     <- input_dim x output_dim 하면 3100이됨. 

# =================================================================
# Total params: 3,100
# Trainable params: 3,100
# Non-trainable params: 0
# _________________________________________________________________

# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 200)            6200   -> output_dim=200이라서

# =================================================================
# Total params: 6,200
# Trainable params: 6,200
# Non-trainable params: 0
# _________________________________________________________________

# model = Sequential()
# model.add(Embedding(input_dim=31, output_dim=200, input_length=5))  # (None, 5, 100)
# model.add(LSTM(10))                                                 # (None, 10)
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 200)            6200

#  lstm (LSTM)                 (None, 10)                8440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 14,761
# Trainable params: 14,761
# Non-trainable params: 0
# _________________________________________________________________

# #3. 컴파일, 훈련
# model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(pad_x, labels, epochs=1000)

# #4. 평가, 예측
# x_predict = ['태운이는 참 재미없다.']  
# token.fit_on_texts(x_predict)
# x_pred = token.texts_to_sequences(x_predict)
# pad_x_pred = pad_sequences(x_pred, maxlen=5)

# # =====================================================
# results = model.evaluate(pad_x, labels)
# final_predict = model.predict(pad_x_pred)

# print('loss: ', results)
# print("[예측]: 태운이는 참 재미없다.", np.round(final_predict))
# loss:  [0.00013755299733020365, 1.0] / [예측]: 태운이는 참 재미없다. [[1.]]

################################## 임베딩 1 #############################
# 정석
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5))
# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 200)            6200

#  lstm (LSTM)                 (None, 10)                8440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

#  embedding_1 (Embedding)     (None, 1, 100)            3100

# =================================================================
# Total params: 17,861
# Trainable params: 17,861
# Non-trainable params: 0
# _________________________________________________________________

################################ 임베딩 2 ####################
# input_length을 지정하지 않아도 자동으로 맞춰줌.
# model.add(Embedding(input_dim=31, output_dim=100))
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 200)            6200

#  lstm (LSTM)                 (None, 10)                8440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

#  embedding_1 (Embedding)     (None, 1, 100)            3100

#  lstm_1 (LSTM)               (None, 10)                4440

#  dense_2 (Dense)             (None, 10)                110

#  dense_3 (Dense)             (None, 10)                110

#  dense_4 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 22,532
# Trainable params: 22,532
# Non-trainable params: 0
# _________________________________________________________________
# model.add(LSTM(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()

################################ 임베딩 3 ####################
# input_dim= 100 수량 변경.
# input_dim= 30 디폴트
# input_dim= 20 # 단어사전의 갯수보다 작을때: 연산량 줄어, 단어사전에서 임의로 빼  : 성능조금 저하
# input_dim= 40 # 단어사전의 갯수보다 클때  : 연산량 늘어, 임의의 랜덤 임베딩 생성 : 성능 조금 저하 

# model = Sequential()
# model.add(Embedding(input_dim=100, output_dim=100))  # (None, 5, 100)
# model.add(LSTM(10))                                                 # (None, 10)
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
# model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(pad_x, labels, epochs=1000)

# #4. 평가, 예측
# results = model.evaluate(pad_x, labels)
# print('loss: ', results)

# loss:  [9.751771722221747e-05, 1.0]
# 인풋딤의 숫자가 바뀌어도 돌아가나 단어사전의 갯수를 명시해줘야지 좋다.
# 왜냐하면 줄이면 성능저하, 너무 많으면 데이터가 과소비된다.

################################ 임베딩 4 ####################
# model.add(Embedding(31, 100)) 따로 명시하지 안하도 돌아감. 라벨값의 갯수, 아웃풋의 갯수
# model.add(Embedding(31, 100, 5))하면 에러난다. 에러명은 아래와 같다.
# ValueError: Could not interpret initializer identifier: 5
# model.add(Embedding(31, 100, input_length=5))해야지 잘 돌아간다.
# model.add(Embedding(31, 100, input_length=6))해서 input_length=6다르면 안됨.
# model.add(Embedding(31, 100, input_length=1))처럼 input_length 1, 5는 돼고, 2,3,4,6... 안돼.

model = Sequential()
model.add(Embedding(3, 100, input_length=1))
model.add(LSTM(10))                                                 # (None, 10)
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 100)         300

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 4,861
# Trainable params: 4,861
# Non-trainable params: 0
# _________________________________________________________________
