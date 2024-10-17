import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print('torch : ', torch.__version__, '사용DEVICE :', DEVICE)

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# 레이블을 0부터 시작하도록 수정 (1을 빼줌)
y = y - 1

# print(x.shape, y.shape) # (581012, 54) (581012,)
# print(np.unique(y))                                     # [0 1 2 3 4 5 6]
# print(f"출력 뉴런 수(클래스 개수): {len(np.unique(y))}") # 출력 뉴런 수(클래스 개수): 7

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
    nn.Linear(54, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 7)
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
print('loss: ', loss)   # loss:  0.3723534345626831

############## acc 출력해봐욤 ##################
y_predict = model(x_test)
print(x_test[:5])
# tensor([[ 1.4159e+00,  1.6378e+00, -1.0819e+00,  1.0291e+00, -4.7012e-01,
#          -9.8689e-01, -1.9228e-01,  4.3871e-01,  5.3473e-01,  1.8245e-01,
#          -9.0233e-01, -2.3314e-01,  1.1373e+00, -2.6065e-01, -7.3145e-02,
#          -1.1522e-01, -9.0783e-02, -1.4734e-01, -5.1909e-02, -1.0698e-01,
#          -1.3208e-02, -1.7074e-02, -4.4365e-02, -2.4375e-01, -1.4735e-01,
#          -2.3292e-01, -1.7523e-01, -3.1937e-02, -2.1424e-03, -6.9721e-02,
#          -7.7550e-02, -5.7440e-02, -8.3873e-02, -1.2705e-01, -3.8412e-02,
#          -2.4748e-01, -3.3179e-01, -1.9462e-01, -2.7861e-02, -6.6421e-02,
#          -4.3658e-02, -4.0767e-02, -4.9722e-01, -2.3409e-01, -2.1518e-01,
#           3.1589e+00, -2.9045e-01, -5.2921e-02, -5.7057e-02, -1.4131e-02,
#          -2.2321e-02, -1.6598e-01, -1.5583e-01, -1.2339e-01],
#         [ 8.1243e-01,  1.6557e+00, -2.8075e-01, -1.0694e+00, -8.4773e-01,
#           1.2010e+00, -6.7784e-01, -6.7381e-02,  6.1313e-01, -6.1746e-02,
#          -9.0233e-01, -2.3314e-01,  1.1373e+00, -2.6065e-01, -7.3145e-02,
#          -1.1522e-01, -9.0783e-02, -1.4734e-01, -5.1909e-02, -1.0698e-01,
#          -1.3208e-02, -1.7074e-02, -4.4365e-02, -2.4375e-01, -1.4735e-01,
#          -2.3292e-01, -1.7523e-01, -3.1937e-02, -2.1424e-03, -6.9721e-02,
#          -7.7550e-02, -5.7440e-02, -8.3873e-02, -1.2705e-01, -3.8412e-02,
#           4.0407e+00, -3.3179e-01, -1.9462e-01, -2.7861e-02, -6.6421e-02,
#          -4.3658e-02, -4.0767e-02, -4.9722e-01, -2.3409e-01, -2.1518e-01,
#          -3.1656e-01, -2.9045e-01, -5.2921e-02, -5.7057e-02, -1.4131e-02,
#          -2.2321e-02, -1.6598e-01, -1.5583e-01, -1.2339e-01],
#         [-3.0517e-01, -1.2485e+00,  5.2043e-01,  1.1373e+00,  1.3150e+00,
#          -8.3043e-01, -4.1639e-01, -1.2314e+00, -3.2771e-01,  5.3173e-01,
#          -9.0233e-01, -2.3314e-01,  1.1373e+00, -2.6065e-01, -7.3145e-02,
#          -1.1522e-01, -9.0783e-02, -1.4734e-01, -5.1909e-02, -1.0698e-01,
#          -1.3208e-02, -1.7074e-02, -4.4365e-02, -2.4375e-01, -1.4735e-01,
#          -2.3292e-01, -1.7523e-01, -3.1937e-02, -2.1424e-03, -6.9721e-02,
#          -7.7550e-02, -5.7440e-02, -8.3873e-02, -1.2705e-01, -3.8412e-02,
#          -2.4748e-01, -3.3179e-01, -1.9462e-01, -2.7861e-02, -6.6421e-02,
#          -4.3658e-02, -4.0767e-02, -4.9722e-01, -2.3409e-01, -2.1518e-01,
#          -3.1656e-01,  3.4429e+00, -5.2921e-02, -5.7057e-02, -1.4131e-02,
#          -2.2321e-02, -1.6598e-01, -1.5583e-01, -1.2339e-01],
#         [ 7.3388e-01, -5.8723e-01, -8.1488e-01,  2.2195e+00, -8.9922e-01,
#           1.2689e+00,  7.4150e-01,  2.3627e-01, -3.5384e-01,  2.5197e-02,
#           1.1082e+00, -2.3314e-01, -8.7930e-01, -2.6065e-01, -7.3145e-02,
#          -1.1522e-01, -9.0783e-02, -1.4734e-01, -5.1909e-02, -1.0698e-01,
#          -1.3208e-02, -1.7074e-02, -4.4365e-02, -2.4375e-01, -1.4735e-01,
#          -2.3292e-01, -1.7523e-01, -3.1937e-02, -2.1424e-03, -6.9721e-02,
#          -7.7550e-02, -5.7440e-02, -8.3873e-02,  7.8710e+00, -3.8412e-02,
#          -2.4748e-01, -3.3179e-01, -1.9462e-01, -2.7861e-02, -6.6421e-02,
#          -4.3658e-02, -4.0767e-02, -4.9722e-01, -2.3409e-01, -2.1518e-01,
#          -3.1656e-01, -2.9045e-01, -5.2921e-02, -5.7057e-02, -1.4131e-02,
#          -2.2321e-02, -1.6598e-01, -1.5583e-01, -1.2339e-01],
#         [-2.6232e-01,  1.5842e+00, -1.4825e+00, -1.2670e+00, -7.9624e-01,
#           1.7665e+00, -5.5245e-03,  5.9054e-01,  4.8246e-01, -5.6375e-01,
#           1.1082e+00, -2.3314e-01, -8.7930e-01, -2.6065e-01, -7.3145e-02,
#          -1.1522e-01, -9.0783e-02, -1.4734e-01, -5.1909e-02, -1.0698e-01,
#          -1.3208e-02, -1.7074e-02, -4.4365e-02, -2.4375e-01, -1.4735e-01,
#          -2.3292e-01, -1.7523e-01, -3.1937e-02, -2.1424e-03, -6.9721e-02,
#          -7.7550e-02, -5.7440e-02, -8.3873e-02, -1.2705e-01, -3.8412e-02,
#          -2.4748e-01,  3.0139e+00, -1.9462e-01, -2.7861e-02, -6.6421e-02,
#          -4.3658e-02, -4.0767e-02, -4.9722e-01, -2.3409e-01, -2.1518e-01,
#          -3.1656e-01, -2.9045e-01, -5.2921e-02, -5.7057e-02, -1.4131e-02,
#          -2.2321e-02, -1.6598e-01, -1.5583e-01, -1.2339e-01]])

y_predict = torch.argmax(model(x_test), dim=1)
print(y_predict[:5])
# tensor([6, 0, 1, 0, 1])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score)) # accuracy : 0.8491
print(f'accuracy : {score:.4f}')         # accuracy : 0.8491

score2 = accuracy_score(y_test.cpu().numpy(),
                        y_predict.cpu().numpy())

# print('accuracy_score : {:4.f}'.format(score2))
print(f'accuracy_score : {score2:.4f}') # accuracy_score : 0.8491

