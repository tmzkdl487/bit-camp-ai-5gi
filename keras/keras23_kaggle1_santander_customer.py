# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

# 맹그러!!!
# 다중분류인줄 알았더니 이진분류였다!!!
# 다중분류 다시 찾겠노라!!!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import time

#1. 데이터
path = 'C://ai5/_data/kaggle/santander-customer-transaction-prediction/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [200000 rows x 201 columns]

test_csv = pd.read_csv(path + "test.csv", index_col = 0)
# print(test_csv) # [200000 rows x 200 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape, test_csv.shape, sample_submission_csv.shape)
# (200000, 201) (200000, 200) (200000, 1)

x  = train_csv.drop(['target'], axis=1) 
# print(x)    #[200000 rows x 200 columns]

y = train_csv['target']
# print(y.shape)  # (200000,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True,
                                                    random_state=3,
                                                    stratify=y)


# print(x_train.shape, y_train.shape) # (100000, 200) (100000,)
# print(x_test.shape, y_test.shape)   # (100000, 200) (100000,)

#2. 모델
model = Sequential()
model.add(Dense(2048, input_dim=200, activation='relu'))
# model.add(Dense(450, activation='relu'))
# model.add(Dense(400))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
# model.add(Dense(250, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))    #'softmax' 'linear'

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])   # 'categorical_crossentropy'

start_time = time.time()

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 100,
    restore_best_weights= True
)

model.fit(x_train, y_train, epochs= 500, batch_size=2000,
          verbose=1, validation_split=0.2, callbacks=[es])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_predict = model.predict(x_test)

y_submit = model.predict(test_csv)

sample_submission_csv['target'] = y_submit

sample_submission_csv.to_csv(path + "sampleSubmission_0724_1630.csv")

print("로스는 : ", round(loss[0], 4))
print("ACC : ", round(loss[1], 3))
print("걸린시간: " , round(end_time - start_time, 2), "초")

# ACC :  0.383
# ACC :  0.9