# m01_pca1.py 복사
# train_test_split 후 스케일링 후 PCA
# 고쳐봐!!!

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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=10, shuffle=True,
    stratify=y, # stratify 정확하게 잘라줌. y의 라벨에 맞춰서 트레인 스플릿 비율을 맞춘다.
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# x_train을 스켈링할 때 fit_transform하고 x_test는 그냥 transfom만 한다.

pca = PCA(n_components=1) 
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#2. 모델
model = RandomForestClassifier(random_state=10)    # 디시전트리 앙상블 모델

#3. 훈련
model.fit(x_train, y_train) # 디폴트 에포 100

#4. 평가, 예측
results = model.score(x_test, y_test) # model.ev가 아니라 스코어로 뽑아줌.

print(x_train.shape, x_test.shape)  # (150, 4)
print('model.score : ', results)

######################### 1점 하기 ######################
# pca 끄고 (150, 4)일때, train_test_split에 랜덤이랑 RandomForestClassifier의 랜덤 바꿔서 만들기
# train_size=0.9하고 랜덤 2개다 10하면 1이 됨.

# pca 끄고 x_train.shape, x_test.shape (135, 4) (15, 4)
# model.score :  1.0

# pca 키고 pca = (135, 3) (15, 3)                                                                                                                                                                                                                                                                                                                                                                                                   
# model.score :  1.0

# pca 키고 (150, 2)일때, (135, 2) (15, 2)
# model.score :  1.0

# pca 키고 (135, 1) (15, 1)일 때,
# model.score :  0.9333333333333333



