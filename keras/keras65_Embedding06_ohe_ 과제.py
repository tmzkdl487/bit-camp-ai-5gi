# 15개의 행에서 5개를 더 넣어서 맹그러
# 예: 반장 주말에 출근 혜지 안혜지 안혜지 // 0

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, Flatten, MaxPooling1D, LSTM, Bidirectional 
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
    '진영이 컴퓨터 샀지 부자지 천재 맞지', 
    '현아 잘 가르쳐주지 착하지', 
    '누리 잘 안 가르쳐주지 안혜지', 
    '혜지 안혜지 웃다가 안혜지', 
    '반장 주말에 출근 혜지 안혜지 안혜지'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0, 1, 1, 0, 0, 0])  # 1이 긍정, 0이 부정

token = Tokenizer() # 토큰 단위로 자른다.
token.fit_on_texts(docs)

# print(token.word_index)
# {'안혜지': 1, '참': 2, '너무': 3, '반장': 4, '잘': 5, '가르쳐주지': 6, '혜지': 7, 
#  '재미있다': 8, '최고에요': 9, '잘만든': 10, '영화예요': 11, '추천하고': 12, '싶은': 13, 
#  '영화입니다': 14, '한': 15, '번': 16, '더': 17, '보고': 18, '싶어요': 19, '글쎄': 20, 
#  '별로에요': 21, '생각보다': 22, '지루해요': 23, '연기가': 24, '어색해요': 25, 
#  '재미없어요': 26, '재미없다': 27, '재밋네요': 28, '준영이': 29, '바보': 30, 
#  '잘생겼다': 31, '태운이': 32, '또': 33, '구라친다': 34, '진영이': 35, '컴퓨터': 36, 
#  '샀지': 37, '부자지': 38, '천재': 39, '맞지': 40, '현아': 41, '착하지': 42, 
#  '누리': 43, '안': 44, '웃다가': 45, '주말에': 46, '출근': 47}

x = token.texts_to_sequences(docs)

# print(x)
# [[3, 8], [2, 9], [2, 10, 11], [12, 13, 14], [15, 16, 17, 18, 19], 
# [20], [21], [22, 23], [24, 25], [26], [3, 27], [2, 28], [29, 30], 
# [4, 31], [32, 33, 34], [35, 36, 37, 38, 39, 40], [41, 5, 6,  42], 
# [43, 5, 44, 6, 1], [7, 1, 45, 1], [4, 46, 47, 7, 1, 1]]

pad_x = pad_sequences(x, maxlen=5)

# print(pad_x.shape)  # (20, 5)

x_train = to_categorical(pad_x)

# print(x_train.shape) # (20, 5, 48)

x_preict = '반장 재미있다.'      # '태운이 참 재미없다'

x_pred = token.texts_to_sequences([x_preict])

pad_x_pred = pad_sequences(x_pred, maxlen=5)

x_pre_ohs = to_categorical(pad_x_pred, num_classes=48)

# print(x_pre_ohs.shape)
# (1, 5, 48) 

# 2. LSTM 모델구성
model = Sequential()
model.add(Bidirectional(LSTM(21, return_sequences=True, input_shape=(5, 31,), activation='relu'))) # timesteps, features
model.add(LSTM(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, labels, epochs=1000,)


#4. 평가, 예측
loss = model.evaluate(x_train, labels)

final_predict = model.predict(x_pre_ohs)

# print("[예측]: 태운이 참 재미없다.", np.round(final_predict))
print("[예측]: 반장 재미있다.", np.round(final_predict))
print("ACC : ", round(loss[1], 3))

# 