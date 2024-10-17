import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'

# text2 = '블라블라블라 많이 많이 해보라고 하셔서 해본다. 일단 쓰라니까 써본다. 변환이 되는지 본다.'

token = Tokenizer()

# 클래스와 인스턴스에 대해서 정리해서 a4에 써서 내기.
token.fit_on_texts([text])

# print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9} <- 많이 나오는 순서, 먼저 나오는 순서.

# print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)]) <- 횟수가 몇 번 

x = token.texts_to_sequences([text])
# print(x)    # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]

# print(x.shape)  # 리스트는 shape 없어!!!! 확인하고 싶으면 랭으로 확인해야됨.

# 원핫인코딩해야된다. 진짜 2랑 매우3 곱하면 맛있는6이 되면 안되니까.

################### 원핫 3가지 맹그러봐!!! ##############
# 위에 데이터를 (14, 9)로 바꿔라. 0빼!
# 케라스, 판다스, 사이킷럿

# 원 핫: 1. 케라스
from tensorflow.keras.utils import to_categorical
x_ohe = np.array(x).reshape(-1)

x_ohe = to_categorical(x)

# print(x_ohe)  # (1, 14, 10)

x_ohe = x_ohe[:, : ,1:]

# print(x_ohe.shape)  # (1, 14, 9)

x_ohe = np.array(x_ohe).reshape(14, 9)

# print(x_ohe.shape)  # (14, 9)

# exit()

##################### 선생님 방법 케라스

# x1 = to_categorical(x, num_classes=10)
# print(x1)
# print(x1.shape)


# 원 핫: 2. 판다스
import pandas as pd

x_flatten = [item for sublist in x for item in sublist] # <- x는 2차원 리스트 형태인데, 이를 1차원 리스트로 바꿔주는 것을 "평탄화(Flatten)"라고 한다.

# print(x_flatten)    # [4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]

x_series = pd.Series(x_flatten) # <- 이 리스트를 pandas.Series로 변환하는 것이다.. Series는 일종의 1차원 배열 같은 것입니다

# print(x_series.shape)  # (14,)

# exit()
x_ohe2 = pd.get_dummies(x_series)
# print(x_ohe2.shape) # (14, 9)


######## 선생님 방법  판다스
x = np.array(x).reshape(-1,)
x3 = pd.get_dummies(x)

print(x3)
print(x.shape)

# 원 핫: 3. 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# x = np.array(x).reshape(-1, 1)
# # print(x.shape)  # (14, 1)
# ohe = OneHotEncoder(sparse=False)    # True가 디폴트
# x_ohe3 = ohe.fit_transform(x) 
# print(x_ohe3.shape) # (14, 9)