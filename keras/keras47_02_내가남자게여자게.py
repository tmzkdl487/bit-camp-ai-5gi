import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

path = 'C:/ai5/_data/image/me/'

x_test = np.load(path + 'me2.npy')

model = load_model('C:/ai5/_data/kaggle/biggest_gender/k45_gender0805_1500_0047-0.347289.hdf5')

y_predict = model.predict(x_test)

print(np.round(y_predict))

# [[0.]] <- 남자로 나옴.