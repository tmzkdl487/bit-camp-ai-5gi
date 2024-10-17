from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # 2중 분류
from sklearn.decomposition import PCA # decomposition 분해; 해체; 부패, 변질

#1. 데이터
datasets = load_iris()

x = datasets['data']
y = datasets.target
# print(x.shape, y.shape) # (150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=1)   # 예시로 3개로하면 컬럼이 3이되고 1개로하면 1개되고 5로하면 에러남.
x = pca.fit_transform(x)

# print(x)    
# # PCA하고 스켈링하는것이 좋음. 반대로 스켈링하고 PCA해도 됨. 의견이 갈림.
# 챗GPT는 스켈링하고 PCA하라고 함.
# 통상 스켈링후 PCA해야 좋아
# [[-2.26470281  0.4800266  -0.12770602]
#  [-2.08096115 -0.67413356 -0.23460885]
#  [-2.36422905 -0.34190802  0.04420148]
#  [-2.29938422 -0.59739451  0.09129011]
#  [-2.38984217  0.64683538  0.0157382 ]
#  [-2.07563095  1.48917752  0.02696829]
# print(x.shape)  # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=10, shuffle=True,
    stratify=y, # stratify 정확하게 잘라줌. y의 라벨에 맞춰서 트레인 스플릿 비율을 맞춘다.
)

#2. 모델
model = RandomForestClassifier(random_state=10)    # 디시전트리 앙상블 모델

#3. 훈련
model.fit(x_train, y_train) # 디폴트 에포 100

#4. 평가, 예측
results = model.score(x_test, y_test) # model.ev가 아니라 스코어로 뽑아줌.

print(x.shape)  # (150, 4)
print('model.score : ', results)

# model.score :  0.9 <- 로스가 아니라 ACC로 뽑아줌. 만점은 1
# 리그레스는 R2 스코러로 뽑아줌.

# model.score :  0.9333333333333333 <- pca 뺀 것 x.shape (150, 4)
# model.score :  0.9                <- pca 3으로하고 x.shape (150, 3)
# model.score :  0.9333333333333333 <- pca 2으로하고 x.shape (150, 2)
# model.score :  0.9333333333333333 <- pca 1으로하고 x.shape (150, 1)
# 따라서, pca 안해서 컬럼 4개랑 2개랑 결과값이 똑같으니 pca 2로하면 자원이 절약된다.

######################### 1점 하기 ######################
# pca 끄고 (150, 4)일때, train_test_split에 랜덤이랑 RandomForestClassifier의 랜덤 바꿔서 만들기
# model.score :  0.9666666666666667 <- 두 개 다 1
# model.score :  0.9666666666666667 <- 2
# model.score :  0.8666666666666667 <- 3
# model.score :  0.9666666666666667 <- 4
# model.score :  0.9333333333333333 <- 5
# model.score :  0.9333333333333333 <- 6
# model.score :  1.0                <- 7

# pca 키고 (150, 3)일때,
# model.score :  0.9333333333333333 <- 두 개 다 1
# model.score :  0.9333333333333333 <- 2
# model.score :  0.8666666666666667 <- 3
# model.score :  0.9666666666666667 <- 4
# model.score :  0.9666666666666667 <- 5
# model.score :  0.9333333333333333 <- 6
# model.score :  0.9333333333333333 <- 7
# model.score :  0.9                <- 8
# model.score :  0.9333333333333333 <- 9
# model.score :  1.0                <- 10

# pca 키고 (150, 2)일때,
# model.score :  0.8333333333333334 <- 두 개 다 1
# model.score :  0.8333333333333334 <- 2
# model.score :  0.8666666666666667 <- 3
# model.score :  1.0                <- 10

# pca 키고 (150, 1)일때, train_size=0.9하고 랜덤 2개다 10하면 1이 됨.
# model.score :  1.0