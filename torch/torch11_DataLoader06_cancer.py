# torch09_class06_cancer.py 카피

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print('torch : ', torch)
# torch :  <module 'torch' from 'c:\\Users\\kim ji hye2\\AppData\\Local\\anaconda3\\envs\\torch241\\Lib\\site-packages\\torch\\__init__.py'>

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, 
                                                    shuffle=True, random_state=369, 
                                                    stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x_train = torch.DoubleTebsor

# int - long
# float - double

x_train = torch.FloatTensor(x_train).to(DEVICE)
# x_train = torch.DoubleTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
# x_test = torch.DoubleTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.DoubleTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.IntTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
# y_test = torch.DoubleTensor(y_test).unsqueeze(1).to(DEVICE)/
# y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)
# y_test = torch.IntTensor(y_test).unsqueeze(1).to(DEVICE)

# print("===========================================")
# print(x_train.shape, x_test.shape)  # torch.Size([398, 30]) torch.Size([171, 30]) 
# print(y_train.shape, y_test.shape)  # torch.Size([398, 1]) torch.Size([171, 1])   
# print(type(x_train), type(y_train)) # <class 'torch.Tensor'> <class 'torch.Tensor'>

from torch.utils.data import TensorDataset  # 텐터 데이터셋 x, y 합친다.
from torch.utils.data import DataLoader     # 데이터로더 batch 정의.

# 토치 데이터셋 만들기 1. x와 y를 합친다.
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
# print(train_set)        # <torch.utils.data.dataset.TensorDataset object at 0x0000023764E6A5B0>
# print(type(train_set))  # <class 'torch.utils.data.dataset.TensorDataset'>
# print(len(train_set))   # 398
# print(train_set[0])     
# (tensor([ 0.1424, -1.2523,  0.2387, -0.0102,  0.4852,  1.4386,  0.6534,  0.3351,
        #  0.9607,  1.6571,  0.5107,  0.5140,  0.9537,  0.2175,  1.0286,  1.4196,
        #  0.6065,  0.6138,  0.7285,  0.6069,  0.0289, -1.1929,  0.1951, -0.1322,
        # -0.0354,  0.7077,  0.2334, -0.0659, -0.0966,  0.5060]), tensor([1.])) <- 튜플형태로 합쳐줌
# print(train_set[0][0])  # 첫번째 x 
# print(train_set[0][0])  # 첫번째 y train_set[397]까지 있겠지.

# 토치 데이터셋 만들기 2. batch를 넣어준다. 끝!!!
train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)
# print(len(train_loader))    # 10 <- 10개로 나눠도 38개 남는데 그 짜투리도 훈련함. 그래서 10개임.
# print(train_loader) # <torch.utils.data.dataloader.DataLoader object at 0x000001E72FE3BF40>
# print(train_loader[0]) # 이터레이터라서 리스트를 볼 때처럼 보려고 하면 에러가 생긴다.

print("==============================================================================")
# 1. 이터레이터를 for문으로 확인.
# for aaa in train_loader:
#     print(aaa)
#     break

# [tensor([[ 0.7500,  0.0157,  0.6547,  ..., -0.4626, -0.5753, -1.3647],
#         [-1.2559, -0.8612, -1.2618,  ..., -0.9542, -0.0743, -0.0912],
#         [ 1.4085,  1.2513,  1.6354,  ...,  0.9229,  0.4602,  1.0968],
#         ...,
#         [ 0.0661, -0.6599,  0.0724,  ...,  0.3932,  0.2640,  0.2493],
#         [ 1.1485,  0.1360,  1.1123,  ...,  1.1704,  0.2799,  0.0204],
#         [-0.3691,  2.2046, -0.4037,  ..., -0.7541, -0.8226, -0.6305]]), tensor([[1.],
#         [1.], 
#         [0.],
#         [1.],
#         [1.],
#         [0.],
#         [1.],
#         [1.],
#         [0.],
#         [1.],
#         [1.],
#         [1.],
#         [1.],
#         [0.],
#         [1.],
#         [1.],
#         [0.],
#         [1.],
#         [0.],
#         [0.],
#         [1.],
#         [1.],
#         [1.],
#         [0.],
#         [0.],
#         [1.],
#         [0.],
#         [1.],
#         [1.],
#         [1.],
#         [0.],
#         [0.],
#         [1.],
#         [1.],
#         [1.],
#         [0.],
#         [1.],
#         [1.],
#         [0.],
#         [1.]])]

# bbb = iter(train_loader)    
# # aaa = bbb.next()  # 파이썬 3.9까지 먹히는 문법으로 아래와 같이 변경해야지 for문이랑 똑같이 나와서 확인할 수 있음.
# aaa = next(bbb) 
# print(aaa)

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.Linear(16, 1),
#     nn.Sigmoid()    
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
    #   super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
     
     # 순전파 !!!   
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x
    
model = Model(30, 1).to(DEVICE)    # 인풋, 아웃풋 정의
    
#3. 컴파일, 훈련
criterion = nn.BCELoss() # 바이너리크로스 엔트로피

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train() # 훈련모드, 디폴트
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
    
        loss.backward()  # 기울기 (gradient)값 계산까지,  # 역전파 시작
        optimizer.step() # 가중치 (w) 갱신                # 역전파 끝
        total_loss += loss.item()
    return total_loss / len(loader)

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss: {}'.format(epochs, loss))   # verbose
    
print("============================")

#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()    # 평가모드로 설정
    total_loss = 0
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_batch, y_predict)
            total_loss += loss2.item()
    return total_loss / len(loader)

# 평가 시 예측 값 생성 및 최종 loss 출력
last_loss = evaluate(model, criterion, test_loader)
print("최종 loss : ", last_loss)

############## 요 밑에 완성할 것 (데이터 로더를 사용하는 것으로 바꿔라) ################

# from sklearn.metrics import accuracy_score

# y_pred = model(x_test)
# # print(y_pred)   #     [1.0000e+00]], grad_fn=<SigmoidBackward0>)

# y_pred =np.round( y_pred.detach().cpu().numpy())
# # print(y_pred)

# y_test = y_test.cpu().numpy()

# acc = accuracy_score(y_test, y_pred)
# print('accuracy: {:.4f}'.format(acc))

# # 최종 loss :  0.522654934095408
# # accuracy: 0.9942

####### 누리 방법 #####################
def acc_score(model, loader):
    x_test = []
    y_test = []
    for x_batch, y_batch in loader:
        x_test.extend(x_batch.detach().cpu().numpy())
        y_test.extend(y_batch.detach().cpu().numpy())
    x_test = torch.FloatTensor(x_test).to(DEVICE)
    y_pre = model(x_test)
    acc = accuracy_score(y_test, np.round(y_pre.detach().cpu().numpy()))
    return acc

acc = acc_score(model, test_loader)
print('acc_score :', acc)

# acc_score : 0.9883040935672515