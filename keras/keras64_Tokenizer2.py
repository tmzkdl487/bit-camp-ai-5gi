# keras64_Tokenizer1.py 복사

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '태운이는 선생을 괴롭힌다. 준영이는 못생겼다. 사영이는 마구 마구 더 못생겼다.'

# 맹그러봐.!!

token = Tokenizer()

token.fit_on_texts([text1, text2])

x = token.texts_to_sequences([text1, text2])

# print(x)    # [[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

x = x[0] + x[1]  

# print(x)    # [5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10, 11, 12, 13, 14, 4, 15, 1, 1, 16, 4]

# 원 핫: 1. 케라스
from tensorflow.keras.utils import to_categorical
x_ohe = np.array(x).reshape(-1)

x1 = to_categorical(x)

# print(x1.shape) # (24, 17)

x_ohe = x1[:, 1:]

# print(x_ohe.shape)  # (24, 16)

# exit()


########### 선생님 케라스
# x1 = to_categorical(x)

# x1 = x1[:, 1:]

# 원 핫: 2. 판다스
# import pandas as pd
# x2 = pd.get_dummies(x)

# print(x2.shape) # (24, 16)

# 원 핫: 3. 사이킷런
from sklearn.preprocessing import OneHotEncoder
x = np.array(x).reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)    # True가 디폴트
x3 = ohe.fit_transform(x)

print(x3.shape) # (24, 16)