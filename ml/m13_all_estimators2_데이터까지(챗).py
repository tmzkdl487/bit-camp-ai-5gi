#  보스톤, 켈리포니아, 디아벳

from sklearn.datasets import load_boston, fetch_california_housing, load_diabetes

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators    # all_estimators 모든 측정기

import sklearn as sk
import time

import warnings
warnings.filterwarnings('ignore')   # 워닝 무시

#1. 데이터
boston = load_boston(return_X_y=True)
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [boston, california, diabetes]
data_name = ['보스톤', '켈리포니아', '디아벳']

#2. 모델구성
all = all_estimators(type_filter='regressor')

kfold = KFold(n_splits=5, shuffle=True, random_state=777)

satrt_time = time.time()

results = []

for index, value in enumerate(datasets): 
       
    x, y = value 
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8,)    # stratify=y  

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    best_model_name = None
    best_model_acc = 0

    for name, model in all:
        try:
            #2. 모델 
            model = model()
            
            #3. 훈련, 평가
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test, y_predict)
            
            if  acc > best_model_acc:
                best_model_acc = acc
                best_model_name = name 
            
            print(name)
        except Exception as e:
            pass
    
    results.append(f"================== {data_name[index]} ====================")
    results.append(f"최고모델 : {best_model_name} {best_model_acc}")
    
    results.append("===========================================================")
end_time = time.time()

# 결과 출력
for result in results:
    print(result)
            
print("걸린시간 : ", round(end_time- satrt_time,2), '초')

# 

