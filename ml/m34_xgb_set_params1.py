from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

import warnings
warnings.filterwarnings('ignore')

#1.  데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=334, train_size=0.8,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters= {
    'n_estimators' : 100,
    'learnin_rate' : 0.1,
    'max_depth' : 5,
}

#2. 모델
model = XGBRFRegressor(random_state=334, **parameters)

model.set_params(gamma=0.4, learning_rate=0.2)

#3. 훈련
model.fit(x_train, y_train,)

#4. 평가, 예측
print("사용파라미터 : ", model.get_params())
results = model.score(x_test, y_test)
print('최종점수 : ', results)

# 사용파라미터 :  {'colsample_bynode': 0.8, 'learning_rate': 0.2, 
# 'reg_lambda': 1e-05, 'subsample': 0.8, 'objective': 'reg:squarederror', 
# 'base_score': None, 'booster': None, 'callbacks': None, 
# 'colsample_bylevel': None, 'colsample_bytree': None, 
# 'device': None, 'early_stopping_rounds': None, 
# 'enable_categorical': False, 'eval_metric': None, 
# 'feature_types': None, 'gamma': 0.4, 'grow_policy': None, 
# 'importance_type': None, 'interaction_constraints': None, 
# 'max_bin': None, 'max_cat_threshold': None, 
# 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 5, 
# 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 
# 'monotone_constraints': None, 'multi_strategy': None, 
# 'n_estimators': 100, 'n_jobs': None, 'num_parallel_tree': None, 
# 'random_state': 334, 'reg_alpha': None, 'sampling_method': None, 
# 'scale_pos_weight': None, 'tree_method': None, 
# 'validate_parameters': None, 'verbosity': None, 'learnin_rate': 0.1}
# 최종점수 :  0.14448722549241932