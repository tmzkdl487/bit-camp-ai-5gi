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
    train_size=0.75, shuffle=True, random_state=1004, stratify=y)

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
print('loss: ', loss)   # loss:  0.5935983657836914

############## acc 출력해봐욤 ##################
y_predict = model(x_test)
print(x_test[:5])
# tensor([[ 0.2095, -0.8182,  0.7721,  0.5505],
#         [ 0.3337, -0.1493,  0.6588,  0.8134],
#         [-1.5286,  0.2966, -1.3232, -1.2899],
#         [-1.4045,  0.2966, -1.3798, -1.2899],
#         [ 1.2027, -0.5953,  0.6022,  0.2876]])

y_predict = torch.argmax(model(x_test), dim=1)
print(y_predict[:5])
# tensor([2, 2, 0, 0, 1])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score)) # accuracy : 0.9211
print(f'accuracy : {score:.4f}')         # accuracy : 0.9211

score2 = accuracy_score(y_test.cpu().numpy(),
                        y_predict.cpu().numpy())

# print('accuracy_score : {:4.f}'.format(score2))
print(f'accuracy_score : {score2:.4f}') # accuracy_score : 0.9211
