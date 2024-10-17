# 02_california
# 03_diabetse

###  요 파일에 이 2개의 데이터셋 다 넣어서 23번처럼 맹그러.

from sklearn.datasets import fetch_california_housing, load_diabetes

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRFRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터셋 로드
data_list = [
    fetch_california_housing(return_X_y=True),
    load_diabetes(return_X_y=True),
]

data_name = ['캘리포니아', "디 아 벳"]

#2. 모델 리스트 정의
model_list = [
    DecisionTreeRegressor(),
    RandomForestRegressor(),
     GradientBoostingRegressor(),
    XGBRFRegressor()
]

for i1, v1 in enumerate(data_list):
    x, y = v1
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=337
    )
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("########################################################")
    print("#######################" , data_name[i1], "#######################")
    print("########################################################")
    
    for i2, v2 in enumerate(model_list):
        if i2 !=3:
            model = v2
        else:
            model = XGBClassifier()
        #3. 훈련
        model.fit(x_train, y_train)
        
        #4. 평가, 예측
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print("========", model.__class__.__name__,"==")
        print("r2_score : ", r2)
        
        if i2 !=3:
            print(model, ":", model.feature_importances_)
        else:
            print("XGBClassifier() : ", model.feature_importances_)       

