# m11_all_estimators1_classifier.py

from sklearn.datasets import load_boston

# 사이킷런 파일 버전 업그레이트해야지 로드 보스턴 파일 사용할 수 있음. 터미널에
# pip install --upgrade scikit-learn==1.1.3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators    # all_estimators 모든 측정기

import sklearn as sk

import warnings
warnings.filterwarnings('ignore')   # 워닝 무시

#1. 데이터
# 데이터 URL
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8,)    #stratify=y

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
all = all_estimators(type_filter='regressor') 

print('all Algorithms : ', all)
print('모델의 갯수: ', len(all))   # 모델의 갯수:  55

# print('sk 버전 : ', sk.__version__) # sk 버전 :  1.5.1

for name, model in all:
    try:
        #2. 모델 
        model = model()
        #3. 훈련
        model.fit(x_train, y_train)
        
        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 : ', acc)
    except:
        print(name, '은 바보 멍충이!!!')
        
# ARDRegression 의 정답률 :  0.6490543159926918
# AdaBoostRegressor 의 정답률 :  0.8384509042261026
# BaggingRegressor 의 정답률 :  0.7997676054865408
# BayesianRidge 의 정답률 :  0.6586373731345692
# CCA 은 바보 멍충이!!!
# DecisionTreeRegressor 의 정답률 :  0.41552642548154994
# DummyRegressor 의 정답률 :  -0.007430492589588278
# ElasticNet 의 정답률 :  0.5731279253295222
# ElasticNetCV 의 정답률 :  0.6569708406332373
# ExtraTreeRegressor 의 정답률 :  0.505465490249108
# ExtraTreesRegressor 의 정답률 :  0.8440016595684765
# GammaRegressor 의 정답률 :  0.6437876396947276
# GaussianProcessRegressor 의 정답률 :  0.1476543393562355
# GradientBoostingRegressor 의 정답률 :  0.811448409351378
# HistGradientBoostingRegressor 의 정답률 :  0.7684401146184295
# HuberRegressor 의 정답률 :  0.6167220417229807
# IsotonicRegression 은 바보 멍충이!!!
# KNeighborsRegressor 의 정답률 :  0.6881140273713647
# KernelRidge 의 정답률 :  -5.440589335485418
# Lars 의 정답률 :  0.6592466510354097
# LarsCV 의 정답률 :  0.6564246721016528
# Lasso 의 정답률 :  0.5596073672171813
# LassoCV 의 정답률 :  0.6568838069466172
# LassoLars 의 정답률 :  -0.007430492589588278
# LassoLarsCV 의 정답률 :  0.6564246721016528
# LassoLarsIC 의 정답률 :  0.6579642926137517
# LinearRegression 의 정답률 :  0.6592466510354096
# LinearSVR 의 정답률 :  0.6118107339281786
# MLPRegressor 의 정답률 :  0.6166298920582728
# MultiOutputRegressor 은 바보 멍충이!!!
# MultiTaskElasticNet 은 바보 멍충이!!!
# MultiTaskElasticNetCV 은 바보 멍충이!!!
# MultiTaskLasso 은 바보 멍충이!!!
# MultiTaskLassoCV 은 바보 멍충이!!!
# NuSVR 의 정답률 :  0.571475022199101
# OrthogonalMatchingPursuit 의 정답률 :  0.4902618098232455
# OrthogonalMatchingPursuitCV 의 정답률 :  0.5895109685394244
# PLSCanonical 은 바보 멍충이!!!
# PLSRegression 의 정답률 :  0.6338953880621446
# PassiveAggressiveRegressor 의 정답률 :  0.24218440884060244
# PoissonRegressor 의 정답률 :  0.7223116318003046
# QuantileRegressor 의 정답률 :  -0.04637643781784373
# RANSACRegressor 의 정답률 :  0.5447136271298729
# RadiusNeighborsRegressor 은 바보 멍충이!!!
# RandomForestRegressor 의 정답률 :  0.7732232713409235
# RegressorChain 은 바보 멍충이!!!
# Ridge 의 정답률 :  0.6591553638389476
# RidgeCV 의 정답률 :  0.6580694089156829
# SGDRegressor 의 정답률 :  0.657110363500492
# SVR 의 정답률 :  0.5895829313193317
# StackingRegressor 은 바보 멍충이!!!
# TheilSenRegressor 의 정답률 :  0.20656646721950156
# TransformedTargetRegressor 의 정답률 :  0.6592466510354096
# TweedieRegressor 의 정답률 :  0.5876808116324395
# VotingRegressor 은 바보 멍충이!!!