import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from bayes_opt import BayesianOptimization
import warnings
import time

warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    shuffle=True,
                                                    random_state=336,
                                                    train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# 3. 검증용 모델 평가 함수
def black_box_function(drop, lr, node1, node2, node3, node4, node5):
    # node 값들을 정수로 변환
    node1, node2, node3, node4, node5 = int(node1), int(node2), int(node3), int(node4), int(node5)
    
    # 모델 빌드
    model = build_model(drop=drop, optimizer='adam', activation='relu',
                        node1=node1, node2=node2, node3=node3, node4=node4, node5=node5, lr=lr)
    
    # 모델 훈련
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0,
              validation_split=0.2, callbacks=[early_stopping])
    
    # 모델 평가
    mae = model.evaluate(x_test, y_test, verbose=0)[1]  # mae 값만 반환
    return -mae  # Bayesian Optimization은 최대화를 시도하므로, mae를 음수로 반환

# 4. Bayesian Optimization을 위한 하이퍼파라미터 범위 설정
pbounds = {
    'drop': (0.2, 0.5),
    'lr': (0.0001, 0.01),
    'node1': (16, 128),
    'node2': (16, 128),
    'node3': (16, 128),
    'node4': (16, 128),
    'node5': (8, 128),
}

# 5. Bayesian Optimization 실행
bay = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=333,
)

# 최적화 실행
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
                         node1=int(best_params['node1']),
                         node2=int(best_params['node2']),
                         node3=int(best_params['node3']),
                         node4=int(best_params['node4']),
                         node5=int(best_params['node5']),
                         lr=best_params['lr'])

# ModelCheckpoint 콜백 추가
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

start_time = time.time()
history = best_model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
                         validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                                                          checkpoint])
end_time = time.time()

# 최종 평가
loss, mae = best_model.evaluate(x_test, y_test)
print("걸린시간 : ", round(end_time - start_time, 2))
print('최종 테스트 손실(MSE):', loss)
print('최종 테스트 MAE:', mae)

# RandomizedSearchCV처럼 추가 정보 출력
print('model.best_params_: ', best_params)
print('model.best_estimator_: ', best_model)
print('model.score: ', best_model.evaluate(x_test, y_test))

# Bayesian Optimization 걸린시간 :  36.38
# 최적 하이퍼파라미터: {'target': -43.80745315551758, 
# 'params': {'drop': 0.2, 
# 'lr': 0.0001, 
# 'node1': 106.79792204976937, 
# 'node2': 54.220108376860814, 
# 'node3': 109.2473166336462, 
# 'node4': 127.56202927976908, 
# 'node5': 80.21385617950648}}

# 걸린시간 :  4.05
# 최종 테스트 손실(MSE): 2940.904541015625
# 최종 테스트 MAE: 43.14929962158203
# model.best_params_:  
# {'drop': 0.2, 'lr': 0.0001, 
# 'node1': 106.79792204976937, 
# 'node2': 54.220108376860814, 
# 'node3': 109.2473166336462, 
# 'node4': 127.56202927976908, 
# 'node5': 80.21385617950648}   
# model.best_estimator_:  <keras.engine.functional.Functional object at 0x000001707A933FD0>
# 3/3 [==============================] - 0s 1ms/step - loss: 2940.9045 - mae: 43.1493
# model.score:  [2940.904541015625, 43.14929962158203]