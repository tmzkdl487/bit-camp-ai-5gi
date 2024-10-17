import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') 

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

print(x_train.shape, y_train.shape) # (60000, 784) (60000,)

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001,):
    inputs = Input(shape=(x_train.shape[1],), name='inputs')
    x = Dense(node1, activation=activation, name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name= 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name= 'hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name= 'hidden4')(x)
    x = Dense(node5, activation=activation, name= 'hidden5')(x)
    outputs = Dense(1, activation='sigmoid', name= 'outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['accuracy'],
                  loss='binary_crossentropy')
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16] 
    node4 = [128, 64, 32, 16] 
    node5 = [128, 64, 32, 16, 8]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            'node4' : node4,
            'node5' : node5,            
            } 
    
hyperparameters = create_hyperparameter()
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

keras_model = KerasClassifier(build_fn=build_model, verbose=1,)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=5,
                           n_iter=10, 
                        #    n_jobs=-1,
                           verbose=1,
                           )

# EarlyStopping 콜백 추가
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=10, restore_best_weights=True)

import time
start_time = time.time()
model.fit(x_train, y_train, 
          epochs=100, 
          validation_split=0.2, callbacks=[early_stopping])
end_time = time.time()

print("걸린시간 : ", round(end_time - start_time, 2))
print('model.best_params_', model.best_params_)
print('model.best_estimator_', model.best_estimator_)
print('model.best_score_', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

# 99712.0000 - val_accuracy: 0.1060
# 걸린시간 :  3343.56
# model.best_params_ {'optimizer': 'adadelta', 'node5': 8, 'node4': 32, 'node3': 64, 'node2': 32, 'node1': 64, 'drop': 0.4, 'batch_size': 500, 'activation': 'linear'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E11932C3D0>
# model.best_score_ 0.1123666673898697
# 20/20 [==============================] - 0s 2ms/step - loss: -2067894915632052293471305728.0000 - accuracy: 0.1135
# model.score :  0.11349999904632568 
