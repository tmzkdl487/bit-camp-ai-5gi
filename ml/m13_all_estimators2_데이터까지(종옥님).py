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

best = []

satrt_time = time.time()

for index, value in enumerate(datasets): 
       
    x, y = value 
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8,)    # stratify=y  

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    maxAccuracy = 0   # 최대의 정확도 / 최고 모델 정의
    maxName = ''      # 최대의 정확도를 가진 모델의 이름

    for name, model in all:
        try:
            #2. 모델 
            model = model()
            
            #3. 훈련, 평가
            scores = cross_val_score(model, x_train, y_train, cv=kfold)

            acc = round(np.mean(scores), 4)
            
            if maxAccuracy < acc:   # acc로 정의한것. 정의 안하고 round(np.mean(scores), 4)) 써도 됨
                maxAccuracy = acc   # 최대의 정확도를 찾아서
                maxName = name      # 모델의 이름을 찾아서    
            
        except:
            print(name, '은 바보 멍충이!!!')
    
        print("======", data_name[index], "======")
        print("======", maxName, maxAccuracy, "======")

            
end_time = time.time()
print("걸린시간 : ", round(end_time- satrt_time,2), '초')

# ====== 보스톤 ======
# ====== ExtraTreesRegressor 0.8838 ======
# 