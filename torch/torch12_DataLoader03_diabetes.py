import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print('torch : ', torch)
# torch :  <module 'torch' from 'c:\\Users\\kim ji hye2\\AppData\\Local\\anaconda3\\envs\\torch241\\Lib\\site-packages\\torch\\__init__.py'>

############ 랜덤 고정하자꾸나 ############################
SEED = 3434

import random
random.seed(SEED)       # 파이썬 랜덤 고정
np.random.seed(SEED)    # 넘파이 랜덤 고정
## 토치 시드 고정
torch.manual_seed(SEED)
## 토치 쿠다 시드 고정
torch.cuda.manual_seed(SEED)

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, 
                                                    shuffle=True, random_state=SEED, 
                                                    # stratify=y
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# torch.Size([309, 10]) torch.Size([133, 10]) torch.Size([309, 1]) torch.Size([133, 1])

from torch.utils.data import TensorDataset  # 텐터 데이터셋 x, y 합친다.
from torch.utils.data import DataLoader     # 데이터로더 batch 정의.

# 토치 데이터셋 만들기 1. x와 y를 합친다.
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# 토치 데이터셋 만들기 2. batch를 넣어준다. 끝!!!
train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)
# print(len(train_loader))    
# print(train_loader) 
# print(train_loader[0]) 

#2. 모델구성
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
        # self.sigmoid = nn.Sigmoid()
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
        # x = self.sigmoid(x)
        return x
    
model = Model(10, 1).to(DEVICE)    # 인풋, 아웃풋 정의
    
#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss() 

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
            hypothesis = model(x_batch)
            loss2 = criterion(hypothesis, y_batch)
            total_loss += loss2.item()
    return total_loss / len(loader)

# 평가 시 예측 값 생성 및 최종 loss 출력
last_loss = evaluate(model, criterion, test_loader)
print("최종 loss : ", last_loss)

def calculate_r2(model, loader):
    x_test = []
    y_test = []
    
    for x_batch, y_batch in loader:
        x_test.extend(x_batch.detach().cpu().numpy())
        y_test.extend(y_batch.detach().cpu().numpy())
        
    x_test = torch.FloatTensor(np.array(x_test)).to(DEVICE)
    
    with torch.no_grad():
        y_pre = model(x_test)
    
    y_pre = y_pre.cpu().numpy()
    
    y_test = np.array(y_test).flatten()
    
    r2 = r2_score(y_test, y_pre)
    return r2

r2 = calculate_r2(model, test_loader)
print('r2 :', r2)

# 최종 loss :  0.0
# r2 : -3.3937949034537738