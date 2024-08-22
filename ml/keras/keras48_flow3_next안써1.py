# keras48_flow2_next.써.py 복사

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
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

augment_size = 100  # 증가시키다.

# print(x_train.shape)    # (60000, 28, 28)

# print(x_train[0])    

# plt.imshow(x_train[0])
# plt.show()

# aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)
# print(aaa.shape)    # (100, 28, 28, 1)

xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),    # x
    np.zeros(augment_size), # y
    batch_size=augment_size,
    shuffle=False,
) # .next()


# print(xy_data)
# print(x_data.shape) # AttributeError: 'tuple' object has no attribute 'shape'
# print(len(xy_data))  # 2 
# print(xy_data[0].shape) # (100, 28, 28, 1)
# print(xy_data[1].shape) # (100,)
# print(type(xy_data))    # .next()가 있으면 <class 'tuple'>
# <class 'keras.preprocessing.image.NumpyArrayIterator'>

# print(len(xy_data)) # 1

# print(xy_data[0].shape) # AttributeError: 'tuple' object has no attribute 'shape'
# print(xy_data[1].shape) # AttributeError: 'tuple' object has no attribute 'shape'

# print(xy_data[0][0].shape)  # (100, 28, 28, 1)


plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)  
    plt.imshow(xy_data[0][0][i], cmap='gray')
    plt.axis('off')

plt.show()
