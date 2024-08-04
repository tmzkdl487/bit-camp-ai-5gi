# keras46_img_to_array.py 복사

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img   #  이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온거 수치화
import matplotlib.pyplot as plt

import numpy as np

path = 'C:/ai5/_data/image/me/me.jpg'

img = load_img(path, target_size=(100, 100),)

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

np.save(np_path + 'me2.npy', arr=img)

################## 요기부터 증폭 ####################

datagen = ImageDataGenerator()

datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.2,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range= 15,      # 정해진 각도만큼 이미지 회전 
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

it = datagen.flow(img,
             batch_size=1,
             )

# print(it)   # <keras.preprocessing.image.NumpyArrayIterator object at 0x0000016F7871BA30>

print(it.next())

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))  # 5개의 미이지를 만듬.

for i in range(5):
    batch = it.next()   # <- datagen이 .next해가지고 5번 돌아감. 그때마다 datagen에 따라 값이 달라짐.
    
    print(batch.shape)   # (1, 100, 100, 3)
    
    batch = (batch.reshape(100, 100, 3))    # 3차원으로 만듬.
    
    ax[i].imshow(batch) 
    ax[i].axis('off')
    
plt.show()
    