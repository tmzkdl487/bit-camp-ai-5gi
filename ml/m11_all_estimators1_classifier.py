# m10_kfold_train_test_split.py

from sklearn.datasets import load_iris

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
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123,
                                                    train_size=0.8, stratify=y,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
all = all_estimators(type_filter='classifier') 

# print('all Algorithms : ', all)
# print('모델의 갯수: ', len(all))    # 모델의 갯수:  43

# all Algorithms :  
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), 
# ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>), 
# ('BernoulliNB', <class 'sklearn.naive_bayes.BernoulliNB'>), 
# ('CalibratedClassifierCV', <class 'sklearn.calibration.CalibratedClassifierCV'>), 
# ('CategoricalNB', <class 'sklearn.naive_bayes.CategoricalNB'>), 
# ('ClassifierChain', <class 'sklearn.multioutput.ClassifierChain'>),
# ('ComplementNB', <class 'sklearn.naive_bayes.ComplementNB'>), 
# ('DecisionTreeClassifier', <class 'sklearn.tree._classes.DecisionTreeClassifier'>), 
# ('DummyClassifier', <class 'sklearn.dummy.DummyClassifier'>), 
# ('ExtraTreeClassifier', <class 'sklearn.tree._classes.ExtraTreeClassifier'>), 
# ('ExtraTreesClassifier', <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>), 
# ('FixedThresholdClassifier', <class 'sklearn.model_selection._classification_threshold.FixedThresholdClassifier'>), 
# ('GaussianNB', <class 'sklearn.naive_bayes.GaussianNB'>), 
# ('GaussianProcessClassifier', <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>), 
# ('GradientBoostingClassifier', <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>), 
# ('HistGradientBoostingClassifier', <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>), 
# ('KNeighborsClassifier', <class 'sklearn.neighbors._classification.KNeighborsClassifier'>), 
# ('LabelPropagation', <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>), 
# ('LabelSpreading', <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>), 
# ('LinearDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>), 
# ('LinearSVC', <class 'sklearn.svm._classes.LinearSVC'>), ('LogisticRegression', <class 'sklearn.linear_model._logistic.LogisticRegression'>), 
# ('LogisticRegressionCV', <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>), ('MLPClassifier', <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>), 
# ('MultiOutputClassifier', <class 'sklearn.multioutput.MultiOutputClassifier'>), ('MultinomialNB', <class 'sklearn.naive_bayes.MultinomialNB'>), 
# ('NearestCentroid', <class 'sklearn.neighbors._nearest_centroid.NearestCentroid'>), ('NuSVC', <class 'sklearn.svm._classes.NuSVC'>), 
# ('OneVsOneClassifier', <class 'sklearn.multiclass.OneVsOneClassifier'>), ('OneVsRestClassifier', <class 'sklearn.multiclass.OneVsRestClassifier'>), 
# ('OutputCodeClassifier', <class 'sklearn.multiclass.OutputCodeClassifier'>), 
# ('PassiveAggressiveClassifier', <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>), 
# ('Perceptron', <class 'sklearn.linear_model._perceptron.Perceptron'>), 
# ('QuadraticDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>), 
# ('RadiusNeighborsClassifier', <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>), 
# ('RandomForestClassifier', <class 'sklearn.ensemble._forest.RandomForestClassifier'>), 
# ('RidgeClassifier', <class 'sklearn.linear_model._ridge.RidgeClassifier'>), 
# ('RidgeClassifierCV', <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>), 
# ('SGDClassifier', <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>), 
# ('SVC', <class 'sklearn.svm._classes.SVC'>), 
# ('StackingClassifier', <class 'sklearn.ensemble._stacking.StackingClassifier'>), 
# ('TunedThresholdClassifierCV', <class 'sklearn.model_selection._classification_threshold.TunedThresholdClassifierCV'>), 
# ('VotingClassifier', <class 'sklearn.ensemble._voting.VotingClassifier'>)]


# all = all_estimators(type_filter='regressor') 
# print('all Algorithms : ', all)
# print('모델의 갯수: ', len(all))    # 모델의 갯수:  43

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
        
# AdaBoostClassifier 의 정답률 :  0.9666666666666667
# BaggingClassifier 의 정답률 :  0.9333333333333333
# BernoulliNB 의 정답률 :  0.6666666666666666
# CalibratedClassifierCV 의 정답률 :  0.9
# CategoricalNB 은 바보 멍충이!!!
# ClassifierChain 은 바보 멍충이!!!
# ComplementNB 은 바보 멍충이!!!
# DecisionTreeClassifier 의 정답률 :  0.8333333333333334
# DummyClassifier 의 정답률 :  0.3333333333333333
# ExtraTreeClassifier 의 정답률 :  0.9333333333333333
# ExtraTreesClassifier 의 정답률 :  0.9
# FixedThresholdClassifier 은 바보 멍충이!!!
# GaussianNB 의 정답률 :  0.9666666666666667
# GaussianProcessClassifier 의 정답률 :  0.9666666666666667
# GradientBoostingClassifier 의 정답률 :  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률 :  0.9333333333333333
# KNeighborsClassifier 의 정답률 :  0.9
# LabelPropagation 의 정답률 :  0.9666666666666667
# LabelSpreading 의 정답률 :  0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률 :  1.0
# LinearSVC 의 정답률 :  0.9666666666666667
# LogisticRegression 의 정답률 :  0.9333333333333333
# LogisticRegressionCV 의 정답률 :  0.9
# MLPClassifier 의 정답률 :  0.9666666666666667
# MultiOutputClassifier 은 바보 멍충이!!!
# MultinomialNB 은 바보 멍충이!!!
# NearestCentroid 의 정답률 :  0.8
# NuSVC 의 정답률 :  0.9333333333333333
# OneVsOneClassifier 은 바보 멍충이!!!
# OneVsRestClassifier 은 바보 멍충이!!!
# OutputCodeClassifier 은 바보 멍충이!!!
# PassiveAggressiveClassifier 의 정답률 :  0.6666666666666666
# Perceptron 의 정답률 :  0.9333333333333333
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9333333333333333
# RadiusNeighborsClassifier 의 정답률 :  0.8666666666666667
# RandomForestClassifier 의 정답률 :  0.9
# RidgeClassifier 의 정답률 :  0.8666666666666667
# RidgeClassifierCV 의 정답률 :  0.8666666666666667
# SGDClassifier 의 정답률 :  0.9333333333333333
# SVC 의 정답률 :  0.9333333333333333
# StackingClassifier 은 바보 멍충이!!!
# TunedThresholdClassifierCV 은 바보 멍충이!!!
# VotingClassifier 은 바보 멍충이!!!

# 선생님이 보여주신 나오는 예시.
# ============== RidgeClassifierCV ====================
# ACC : [0.625      0.875     0.8333333   0.875     0.83333333]
# 평균 ACC : 0.8083
# cross_val_predict ACC : 0.8