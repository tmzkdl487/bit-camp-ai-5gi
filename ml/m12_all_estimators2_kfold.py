from sklearn.datasets import load_boston

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

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

for name, model in all:
    try:
        #2. 모델 
        model = model()
        
        #3. 훈련
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print("========================", name, "=================")
        print("ACC : ", scores, '\n, 평균ACC : ', round(np.mean(scores), 4))
        
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predict)
        print('cross_val_predict ACC : ', acc)
        
    except:
        print(name, '은 바보 멍충이!!!')


# ARDRegression 은 바보 멍충이!!!
# AdaBoostRegressor 은 바보 멍충이!!!
# BaggingRegressor 은 바보 멍충이!!!
# BayesianRidge 은 바보 멍충이!!!
# CCA 은 바보 멍충이!!!
# DecisionTreeRegressor 은 바보 멍충이!!!
# DummyRegressor 은 바보 멍충이!!!
# ElasticNet 은 바보 멍충이!!!
# ElasticNetCV 은 바보 멍충이!!!
# ExtraTreeRegressor 은 바보 멍충이!!!
# ExtraTreesRegressor 은 바보 멍충이!!!
# GammaRegressor 은 바보 멍충이!!!
# GaussianProcessRegressor 은 바보 멍충이!!!
# GradientBoostingRegressor 은 바보 멍충이!!!
# HistGradientBoostingRegressor 은 바보 멍충이!!!
# HuberRegressor 은 바보 멍충이!!!
# IsotonicRegression 은 바보 멍충이!!!
# KNeighborsRegressor 은 바보 멍충이!!!
# KernelRidge 은 바보 멍충이!!!
# Lars 은 바보 멍충이!!!
# LarsCV 은 바보 멍충이!!!
# Lasso 은 바보 멍충이!!!
# LassoCV 은 바보 멍충이!!!
# LassoLars 은 바보 멍충이!!!
# LassoLarsCV 은 바보 멍충이!!!
# LassoLarsIC 은 바보 멍충이!!!
# LinearRegression 은 바보 멍충이!!!
# LinearSVR 은 바보 멍충이!!!
# MLPRegressor 은 바보 멍충이!!!
# MultiOutputRegressor 은 바보 멍충이!!!
# MultiTaskElasticNet 은 바보 멍충이!!!
# MultiTaskElasticNetCV 은 바보 멍충이!!!
# MultiTaskLasso 은 바보 멍충이!!!
# MultiTaskLassoCV 은 바보 멍충이!!!
# NuSVR 은 바보 멍충이!!!
# OrthogonalMatchingPursuit 은 바보 멍충이!!!
# OrthogonalMatchingPursuitCV 은 바보 멍충이!!!
# PLSCanonical 은 바보 멍충이!!!
# PLSRegression 은 바보 멍충이!!!
# PassiveAggressiveRegressor 은 바보 멍충이!!!
# PoissonRegressor 은 바보 멍충이!!!
# QuantileRegressor 은 바보 멍충이!!!
# RANSACRegressor 은 바보 멍충이!!!
# RadiusNeighborsRegressor 은 바보 멍충이!!!
# RandomForestRegressor 은 바보 멍충이!!!
# RegressorChain 은 바보 멍충이!!!
# Ridge 은 바보 멍충이!!!
# RidgeCV 은 바보 멍충이!!!
# SGDRegressor 은 바보 멍충이!!!
# SVR 은 바보 멍충이!!!
# StackingRegressor 은 바보 멍충이!!!
# TheilSenRegressor 은 바보 멍충이!!!
# TransformedTargetRegressor 은 바보 멍충이!!!
# TweedieRegressor 은 바보 멍충이!!!
# VotingRegressor 은 바보 멍충이!!!