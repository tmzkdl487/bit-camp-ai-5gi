# torch07_다중_00_iris.py 카피

# 00_iris
# 06_cancer.py         // 이진
# 07_dacon_diabetes.py // 이진
# 08_kaggle_bank.py    // 이진
# 09_wine.py
# 10_fetch_covtype.py
# 11_digits.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print('torch : ', torch.__version__, '사용DEVICE :', DEVICE)

############ 랜덤 고정하자꾸나 ############################
SEED = 0

import random
random.seed(SEED)       # 파이썬 랜덤 고정
np.random.seed(SEED)    # 넘파이 랜덤 고정
## 토치 시드 고정
torch.manual_seed(SEED)
## 토치 쿠다 시드 고정
torch.cuda.manual_seed(SEED)

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (150, 4) (150,)

# y의 고유값 확인
# print(np.unique(y))                                     # [0 1 2]
# print(f"출력 뉴런 수(클래스 개수): {len(np.unique(y))}") # 출력 뉴런 수(클래스 개수): 3

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.75, shuffle=True, random_state=SEED, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

#2. 모델
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()   # 분류 문제에 적합한 손실 함수

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    # model.train*()
    optimizer.zero_grad()   
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward() # 역전파 기중치 계산, 가중치 갱신 
    optimizer.step()
    return loss.item()

EPOCHS = 1000   # 특정 상수를 표시하려면 대문자로 쓸 때가 있음.
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    # print('epoch : {}, loss: {:.8f}'.format(epoch, loss))
    print(f'epoch : {epoch}, loss: {loss:.8f}')
    
#4. 평가, 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        
        return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
print('loss: ', loss)   # loss:  0.014776838012039661

############## acc 출력해봐욤 ##################
y_predict = model(x_test)
print(x_test[:5])
# tensor([[-0.8943,  0.7983, -1.2714, -1.3276],
#         [-1.2445, -0.0869, -1.3274, -1.4591],
#         [-0.6608,  1.4622, -1.2714, -1.3276],
#         [-0.8943,  0.5770, -1.1594, -0.9332],
#         [-0.4273, -1.4148, -0.0395, -0.2759]])

y_predict = torch.argmax(model(x_test), dim=1)
print(y_predict[:5])
# tensor([0, 0, 0, 0, 1])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score)) # accuracy : 1.0000
print(f'accuracy : {score:.4f}')         # accuracy : 1.0000

score2 = accuracy_score(y_test.cpu().numpy(),
                        y_predict.cpu().numpy())

# print('accuracy_score : {:4.f}'.format(score2))
print(f'accuracy_score : {score2:.4f}') # accuracy_score : 1.0000

