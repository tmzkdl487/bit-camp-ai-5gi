# https://autokeras.com/ 참고라고 선배님 깃허브에서 발견.
# pip3 install autokeras
# 참고: 현재 AutoKeras는 Python >= 3.7 및 TensorFlow >= 2.8.0 와만 호환됩니다 .

# print(ak.__version__)   # 1.0.15
# print(tf.__version__)   # 2.7.4 -> 2.15.1
# print(keras.__version__)    # 2.7.0
# 에러났음.

# 설치 최종 버전
# pip install ak3 autokeras==1.0.20 tensorflow-gpu==2.10.1

# from tensorflow.keras.layers import preprocessing
import autokeras as ak
import tensorflow as tf
import keras

import time

# print(ak.__version__)   # 1.0.20
# print(tf.__version__)   # 2.10.1
# print(keras.__version__)    # 2.10.0
# cuda 12.5 / cudnn X

# TensorFlow에서 EagerExecution 비활성화 (필요시 사용)
tf.compat.v1.disable_eager_execution()  

is_eager = tf.executing_eagerly()
print("EagerExecution 활성화 여부:", is_eager)  # EagerExecution 활성화 여부: False

#1. 데이터
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

#2. 모델  
model = ak.ImageClassifier(
    overwrite=False, # 문제가 되는 부분을 해결하기 위해 overwrite=True로 변경
    max_trials=3,
)

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train, epochs=1, validation_split=0.15)
end_time = time.time()

##### 최적의 출력 모델 ######
best_model = model.export_model()
print(best_model.summary())

##### 최적의 모델 저장 ######
path = '.\\_save\\autokeras\\'                                                                                                                                                                                                                                    
best_model.save(path + 'keras70_autokeras1.h5') # model.export_model() <- export 수출하다

#4. 평가, 예측 
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('model 결과 : ', results)

y_predict2 = best_model.predict(x_test)
# results2 = best_model.evaluate(x_test, y_test)
# print('best_model 결과 : ', results)

# print('걸린시간 : ', round(end_time - start_time, 2), '초')

# TypeError: Unable to serialize [2.0896919 2.1128857 2.1081853] to JSON. Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>.
# 사영님 돌렸는데 10시간 걸렸다고 함. / 사영님: ACC 0.989 나옴. / 태운님: 97.5 나옴. 