from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators    # all_estimators 모든 측정기

import sklearn as sk
import time

import warnings
warnings.filterwarnings('ignore')   # 워닝 무시

#1. 데이터
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digit = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digit]
data_name = ['아이리스', '캔서', '와인', '디지트']

#2. 모델구성
all = all_estimators(type_filter='classifier')

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

satrt_time = time.time()

results = []

for index, value in enumerate(datasets): 
       
    x, y = value 
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8, stratify=y,)

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
            # print(f"모델 {name} 실행 오류: {e}")
    
    results.append(f"================== {data_name[index]} ====================")
    results.append(f"최고모델 : {best_model_name} {best_model_acc}")
    
    results.append("===========================================================")
end_time = time.time()

# 결과 출력
for result in results:
    print(result)
            
print("걸린시간 : ", round(end_time- satrt_time,2), '초')

# 수정되기 전에 모델 돌렸을때 나온 시간. 너무 많이 모델이 나와서 복사 안함.
# 걸린시간 :  95.22 초

# 선생님이 보여주신 예시
# ========= 아이리스 =========
# 최고모델 : LinearDiscriminantAnalysis 0.98
# =============================
# ========= 캔서  =========
# 최고모델 : AistGradientBoostingClassifier 0.9666
# =============================
# ========= 와인  =========
# 최고모델 : QuadraticDiscriminantAnalysis 0.9889
# =============================
# ========= 디지트  =========
# 최고모델 : SVC 0.9866
# ============================


# 내가 만든 모델 출력된 예시!
# ================== 아이리스 ====================
# 최고모델 : LinearDiscriminantAnalysis 1.0
# ===========================================================
# ================== 캔서 ====================
# 최고모델 : RidgeClassifierCV 0.9736842105263158
# ===========================================================
# ================== 와인 ====================
# 최고모델 : GaussianNB 1.0
# ===========================================================
# ================== 디지트 ====================
# 최고모델 : ExtraTreesClassifier 0.9583333333333334
# ===========================================================
# 걸린시간 :  93.72 초

