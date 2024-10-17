import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스켈링한 데이터로 줘라, 수치화. 수치화만 하고 싶으면 밑에는 다 안써도 됨.
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평행이동 <- 위에 수평, 수직, 평행이동 데이터를 추가하면 8배의 데이터가 늘어난다.
    height_shift_range=0.1, # 평행이동 수직
    rotation_range= 5,      # 정해진 각도만큼 이미지 회전 
    zoom_range=1.2,         # 축소 또는 확대
    shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환.
    fill_mode='nearest',    # 몇 개 더 있지만, 대표적으로 0도 있음. 너의 빈자리 비슷한 거로 채워줄께.
)

test_datagen = ImageDataGenerator(
    rescale=1./255) # 평가 데이터므로 리쉐이프하고 절.대.로 변환하지않고 수치화만 한다.

path_train = './_data/image/brain/train/' # 이미지 데이터의 상위 폴더만 적으면 나머지 폴더 2개의 데이터들은 각각 0, 1로 바뀐다.
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, # 트레인 폴더에 있는 것을 수치화해줘라.
    target_size=(200, 200),  # 타겟 사이즈를 200에 200으로 잡는다.
    batch_size=10,  # 10, 200, 200, 1로 
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)   # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(200, 200),  # 타겟 사이즈를 200에 200으로 잡는다.
    batch_size=10,  # 10, 200, 200, 1로 
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True, # test 데이터는 shuffle 하지 않음
)   # Found 120 images belonging to 2 classes.




