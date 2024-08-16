from keras.preprocessing.text import Tokenizer
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

# print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, 
# '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, 
# '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, 
# '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '준영이': 24, 
# '바보': 25, '반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
# print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], 
# [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]

# print(type(x)) # <class 'list'> 

# print(y)    # [[28, 1, 22]] 

from tensorflow.keras.preprocessing.sequence import pad_sequences

# 맹그러봐. (15, 5)

pad_x = pad_sequences(x,
                      # padding = 'post', # padding = 'pre'(앞에), 'post'(뒤에 0이 생김)/ 안써도 디폴트로 앞에 0을 넣어줌.
                      maxlen=4, # 자르는 갯수
                      # truncating= 'pre' # <- 앞에서부터 자르기, 뒤에서 부터 자르기
                    ) 
# print(pad_x) # <- 앞에 0 채워졌는지 뒤에 0이 채워지는지 볼 것.
# print(pad_x.shape)  # (15, 5)

# print(pad_y)    # [[ 0  0 28  1 22]]
# print(pad_y.shape)  # (1, 5)

#2. 모델
######################### DNN 맹그러봐 ###########################
# 예측: 태운이는 참 재미없다.
# AAC: 1.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import time

x_train, x_test, y_train, y_test = train_test_split(pad_x, lavels, train_size=0.9, random_state=8989)

# scaler = MaxAbsScaler() 

model = Sequential()
model.add(Dense(20, activation='relu', input_dim=5))   # relu는 음수는 무조껀 0으로 만들어 준다.
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(pad_x, labels, epochs=1000,)


#4. 평가, 예측
loss = model.evaluate(pad_x, labels)

x_predict = ['태운이 참 재미없다']
token.fit_on_texts(x_predict)
x_predict2 = token.texts_to_sequences(x_predict)
pad_x2 = pad_sequences(x_predict2, maxlen=5) # padding='pre'
final_predict = model.predict(pad_x2)

print("[예측]: 태운이 참 재미없다", np.round(final_predict)) 

# y_predict = ['태운이는 참 재미있다.']
# token = Tokenizer()
# token.fit_on_texts(y_predict)
# y_predict = token.texts_to_sequences(y_predict)
# pad_y2 = pad_sequences(y_predict, padding='pre', maxlen=5) 

# final_predict2 = model.predict(pad_y2)
 
print("ACC : ", round(loss[1], 3))
# print("[예측]: 태운이 참 재미있다.", np.round(final_predict2))

# 예측: 태운이 참 재미없다 / AAC: 1.0 만들 것.

# [예측]: 태운이 참 재미없다 [[1.]] / ACC :  0.867
