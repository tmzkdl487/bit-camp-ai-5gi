import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from bayes_opt import BayesianOptimization
import warnings
import time

warnings.filterwarnings('ignore')

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# CNN을 위한 4D 텐서 형태로 데이터 변환
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)
x_test = scaler.transform(x_test.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)

# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                filters1=32, filters2=64, node=128, kernel_size=3, lr=0.001):
    inputs = Input(shape=(28, 28, 1), name='inputs')
    x = Conv2D(filters=filters1, kernel_size=(kernel_size, kernel_size), activation=activation, name='conv1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=filters2, kernel_size=(kernel_size, kernel_size), activation=activation, name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(node, activation=activation, name='dense')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. 검증용 모델 평가 함수
def black_box_function(drop, lr, filters1, filters2, node, kernel_size):
    filters1, filters2, node = int(filters1), int(filters2), int(node)
    kernel_size = int(kernel_size)
    
    model = build_model(drop=drop, optimizer='adam', activation='relu',
                        filters1=filters1, filters2=filters2, node=node, kernel_size=kernel_size, lr=lr)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0,
              validation_split=0.2, callbacks=[early_stopping])
    
    acc = model.evaluate(x_test, y_test, verbose=0)[1]  # accuracy 값만 반환
    return acc  # Bayesian Optimization은 최대화를 시도하므로 accuracy 그대로 반환

# 4. Bayesian Optimization을 위한 하이퍼파라미터 범위 설정
pbounds = {
    'drop': (0.2, 0.5),
    'lr': (0.0001, 0.01),
    'filters1': (16, 64),
    'filters2': (16, 64),
    'node': (64, 256),
    'kernel_size': (2, 5),
}

# 5. Bayesian Optimization 실행
bay = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=333,
)

n_iter = 10  # 예시로 10회 실행
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print("Bayesian Optimization 걸린시간 : ", round(end_time - start_time, 2))

# 최적 하이퍼파라미터 출력
print('최적 하이퍼파라미터:', bay.max)

# 최적화된 모델로 최종 평가
best_params = bay.max['params']
best_model = build_model(drop=best_params['drop'],
                         optimizer='adam',
                         activation='relu',
                         filters1=int(best_params['filters1']),
                         filters2=int(best_params['filters2']),
                         node=int(best_params['node']),
                         kernel_size=int(best_params['kernel_size']),
                         lr=best_params['lr'])

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

start_time = time.time()
history = best_model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
                         validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                                                          checkpoint])
end_time = time.time()

# 최종 평가
loss, accuracy = best_model.evaluate(x_test, y_test)
print("걸린시간 : ", round(end_time - start_time, 2))
print('최종 테스트 손실:', loss)
print('최종 테스트 정확도:', accuracy)

# 추가 정보 출력
print('model.best_params_: ', best_params)
print('model.best_estimator_: ', best_model)
print('model.score: ', best_model.evaluate(x_test, y_test))

# Bayesian Optimization 걸린시간 :  661.45
# 최적 하이퍼파라미터: {'target': 0.9927999973297119, 
# 'params': {'drop': 0.4302753439899352, 
# 'filters1': 32.94381238992843, 
# 'filters2': 24.449099975092057, 
# 'kernel_size': 4.006469458917124, 
# 'lr': 0.004242082442754971, 
# 'node': 144.3741438714784}}

# 걸린시간 :  265.96
# 최종 테스트 손실: 0.02271946705877781
# 최종 테스트 정확도: 0.9934999942779541
# model.best_params_:  {'drop': 0.4302753439899352, 
# 'filters1': 32.94381238992843, 
# 'filters2': 24.449099975092057, 
# 'kernel_size': 4.006469458917124, 
# 'lr': 0.004242082442754971, 
# 'node': 144.3741438714784}
# model.best_estimator_:  <keras.engine.functional.Functional object at 0x000001D79B6C7BE0>
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0227 - accuracy: 0.9935
# model.score:  [0.02271946705877781, 0.9934999942779541]