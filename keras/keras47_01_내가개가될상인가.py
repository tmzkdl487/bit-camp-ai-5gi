from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

path = 'C:/ai5/_data/image/me/me.npy'

me = np.load(path)

print(me.shape)

model = load_model('C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/k35_040804_2156_0016-0.563462.hdf5')

y_predict = model.predict(me)

loss = 1-y_predict

print(y_predict)
print(np.round(y_predict))
print('나는 {}확률로 고양이입니다.'.format(loss))

# print(xy_tarin.class_indices)   # 데이터 라벨 종류 알아내기 -> 개가 0인지 1인지 알아내려고 보는 것. 여기서 안나오고 캣앤독 데이터가야지 알수 있음.

# 0은 개 /  1은 고양이
# [[1.]] -> 나는 고양이 상으로 나옴