import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

############ 랜덤 고정하자꾸나 ############################
SEED = 18

import random
random.seed(SEED)       # 파이썬 랜덤 고정
np.random.seed(SEED)    # 넘파이 랜덤 고정
## 토치 시드 고정
torch.manual_seed(SEED)
## 토치 쿠다 시드 고정
torch.cuda.manual_seed(SEED)

#1. 데이터
path = 'C://ai5/_data/dacon/diabetes/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)    # [652 rows x 9 columns]

test_csv = pd.read_csv(path + "test.csv", index_col = 0)
# print(test_csv) # [116 rows x 8 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape, test_csv.shape, sample_submission_csv.shape) 
# # (652, 9) (116, 8) (116, 1)

# print(train_csv.info())

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=SEED,
                                                    stratify=y
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# torch.Size([586, 8]) torch.Size([66, 8]) torch.Size([586, 1]) torch.Size([66, 1])

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
    
model = Model(8, 2).to(DEVICE)    # 인풋, 아웃풋 정의
    
#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss() 

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train() # 훈련모드, 디폴트
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        
        # 차원 줄이기
        y_batch = y_batch.squeeze(1)  # [batch_size, 1] -> [batch_size]
        
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
            
            # 차원 줄이기
            y_batch = y_batch.squeeze(1)  # [batch_size, 1] -> [batch_size]
            
            loss2 = criterion(hypothesis, y_batch)
            total_loss += loss2.item()
    return total_loss / len(loader)

# 평가 시 예측 값 생성 및 최종 loss 출력
last_loss = evaluate(model, criterion, test_loader)
print("최종 loss : ", last_loss)

def acc_score(model, loader):
    x_test = []
    y_test = []
    
    for x_batch, y_batch in loader:
        x_test.extend(x_batch.detach().cpu().numpy())
        y_test.extend(y_batch.detach().cpu().numpy())
        
    x_test = torch.FloatTensor(np.array(x_test)).to(DEVICE)
    
    with torch.no_grad():
        y_pre = model(x_test)
    
    y_pre = torch.argmax(y_pre, dim=1).cpu().numpy()
    
    y_test = np.array(y_test).flatten()
    
    acc = accuracy_score(y_test, y_pre)
    return acc

acc = acc_score(model, test_loader)
print('acc_score :', acc)

# 최종 loss :  6.384978771209717
# acc_score : 0.7424242424242424



