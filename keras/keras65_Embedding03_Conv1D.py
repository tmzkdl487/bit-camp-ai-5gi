# 땡겨서 맹그러!!!!
# 지표는 ACC
# 예측: 태운이는 참 재미없다.

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, Flatten, MaxPooling1D, LSTM 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

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

x_predict = ['태운이는 참 재미없다.']
token = Tokenizer()
token.fit_on_texts(x_predict)
x_predict = token.texts_to_sequences(x_predict)
pad_x2 = pad_sequences(x_predict, maxlen=5) # padding='pre'

# print(pad_x2.shape) # (1, 5)


pad_x = pad_x.reshape(15, 5, 1) 

# print(pad_x.shape)  # (15, 5, 1)

pad_x2  = pad_x2.reshape(1, 5, 1)

# print(pad_x2.shape) # (1, 5, 1)

#2. Conv1D 모델
model = Sequential()
model.add(Conv1D(10, kernel_size=2, input_shape=(5, 1))) # timesteps, features
model.add(Conv1D(10, 2))
model.add(Flatten())
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(pad_x, labels, epochs=1000,)


#4. 평가, 예측
loss = model.evaluate(pad_x, labels)

final_predict = model.predict(pad_x2)


print("[예측]: 태운이는 참 재미없다.", np.round(final_predict))
print("ACC : ", round(loss[1], 3))

# [예측]: 태운이는 참 재미없다. [[1.]]
# ACC :  1.0