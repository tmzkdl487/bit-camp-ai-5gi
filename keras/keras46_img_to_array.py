from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img   #  이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온거 수치화
import matplotlib.pyplot as plt

import numpy as np

path = 'C:/ai5/_data/image/me/me.jpg'

img = load_img(path, target_size=(80, 80),)

print(img)
# <PIL.Image.Image image mode=RGB size=200x200 at 0x1E4B13DA6A0>

print(type(img))
# plt.imshow(img)
# plt.show()  # 사진을 볼 수 있다.

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (200, 200, 3)
print(type(arr))    # <class 'numpy.ndarray'>

# 차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape)    # (1, 100, 100, 3)

# me 폴더에 위에 데이커를 npy로 저장할 것
np_path = 'C:/ai5/_data/image/me/'

np.save(np_path + 'me3.npy', arr=img)
