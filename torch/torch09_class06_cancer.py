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

print("===========================================")
print(x_train.shape, x_test.shape)  # torch.Size([398, 30]) torch.Size([171, 30]) 
print(y_train.shape, y_test.shape)  # torch.Size([398, 1]) torch.Size([171, 1])   
print(type(x_train), type(y_train)) # <class 'torch.Tensor'> <class 'torch.Tensor'>

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
# loss = model.evaluate(x, y)
# def evaluate(model, criterion, x, y):
#     model.eval()    # 평가모드 // 역전파, 가중치 갱신, 기울기 계산할 수 있기도 없기도,
#                     # 드롭아웃, 배치모멀 <- 얘네들 몽땅 하지마!!!
#     with torch.no_grad():
#         y_predict = model(x)
#         loss2 = criterion(y, y_predict)
#     return loss2.item()

# last_loss = evaluate(model, criterion, x_test, y_test)
# print("최종 loss : ", last_loss)    # 최종 loss :  1.1714887619018555

############################## 요 밑에 완성할 것 #################
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

# 예측 값 0.5 기준으로 0 또는 1로 변환 / 부동 소수점 형태로 변환해서 정확한 계산을 하도록 함.
# y_predict_class = (y_predict >= 0.5).float()

# CPU로 이동하여 numpy 변환 (torch tensor를 sklearn에 맞게 변환)
# y_predict_class = y_predict_class.cpu().numpy()
# y_test_cpu = y_test.cpu().numpy()

from sklearn.metrics import accuracy_score
# 정확도 계산
# accuracy = accuracy_score(y_test_cpu, y_predict_class)
# print("최종 accuracy : ", accuracy)

# 최종 loss :  0.16224685311317444
# 최종 accuracy :  0.9824561403508771

########################### 사영님 방법 ############################

#y_test = torch.round(y_test)
#y_pred = torch.round(y_pred)

#y_test = y_test.detach().cpu().numpy()
#y_pred = y_pred.detach().cpu().numpy()

#acc = accuracy_score(y_test, y_pred)
#print("최종 accuracy : ", acc)

######################### 선생님 방법 ###################################
y_pred = model(x_test)
# print(y_pred)   #     [1.0000e+00]], grad_fn=<SigmoidBackward0>)

y_pred =np.round( y_pred.detach().cpu().numpy())
# print(y_pred)

y_test = y_test.cpu().numpy()

acc = accuracy_score(y_test, y_pred)
print('accuracy: {:.4f}'.format(acc))

# 최종 loss :  0.635149359703064
# accuracy: 0.9883