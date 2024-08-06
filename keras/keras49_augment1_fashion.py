# keras48_flow2_next.써.py 복사

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

train_datagen = ImageDataGenerator(
    # rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.2,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range= 15,      # 정해진 각도만큼 이미지 회전 
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

augment_size = 40000  # 증가시키다.

# print(x_train.shape[0]) # 60000

randidx = np.random.randint(x_train.shape[0], size=augment_size)    # 6000, size=40000
# print(randidx)  # [50310 52873 55776 ...  9744 16844 21876] -> 4만개를 뽑았음.

# print(np.min(randidx), np.max(randidx))  # 7 59997

# print(x_train[0].shape) # (28, 28)

x_augmented = x_train[randidx].copy()   # .copy()하면 메모리값을 새로 할당하기 때문에 원래 메모리값에 영향을 미치지 않는다. 메모리 안전빵.
y_augmented = y_train[randidx].copy()   # x, y 4만개 준비됨.

# print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],           # 40000
    x_augmented.shape[1],           # 28
    x_augmented.shape[2], 1)        # 28, 1

# print(x_augmented.shape)    # (40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape) # (40000, 28, 28, 1) <- train_datagen = ImageDataGenerator해서 변형된 4만개 데이터 생성 끝!

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

# 넘파이의 행렬 데이터를 합치는 것 찾아보기.

x_train = np.concatenate((x_train, x_augmented))   
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)    #  (100000, 28, 28, 1) (100000,)

# 맹그러봐