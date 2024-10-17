import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# 1. 데이터 경로 설정
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

# 데이터 로드
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# 범주형 데이터 인코딩
encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

# 학습에 사용할 데이터와 타깃 설정
x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']

# 테스트 데이터 처리
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 데이터 분할 (80% 학습, 20% 테스트)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    shuffle=True, random_state=666)

# Numpy 배열로 변환
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()  # y_train과 y_test도 Numpy 배열로 변환
y_test = y_test.to_numpy()

# 데이터 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# PyTorch 텐서로 변환
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# 출력 형태 확인
print("===========================================")
print(x_train.shape, x_test.shape)  #  torch.Size([132027, 10]) torch.Size([33007, 10])
print(y_train.shape, y_test.shape)  #  torch.Size([132027, 1]) torch.Size([33007, 1])
print(type(x_train), type(y_train)) # <class 'torch.Tensor'> <class 'torch.Tensor'>

#2. 모델구성
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 1),
    nn.Sigmoid()    
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss() # 바이너리크로스 엔트로피

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train() # 훈련모드, 디폴트
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()  # 기울기 (gradient)값 계산까지,  # 역전파 시작
    optimizer.step() # 가중치 (w) 갱신                # 역전파 끝
    return loss.item()

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epochs, loss))   # verbose
    
print("============================")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()    # 평가모드로 설정
    with torch.no_grad():
        y_predict = model(x)
        loss = criterion(y_predict, y)
        return loss.item(), y_predict

# 평가 시 예측 값 생성 및 최종 loss 출력
last_loss, y_predict = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", last_loss)

y_pred = model(x_test)
# print(y_pred)   #     [1.0000e+00]], grad_fn=<SigmoidBackward0>)

y_pred =np.round( y_pred.detach().cpu().numpy())
# print(y_pred)

y_test = y_test.cpu().numpy()

acc = accuracy_score(y_test, y_pred)
print('accuracy: {:.4f}'.format(acc))

# 최종 loss :  0.3236919939517975
# accuracy: 0.8635