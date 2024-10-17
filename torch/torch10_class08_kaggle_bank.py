import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print('torch : ', torch)
# torch :  <module 'torch' from 'c:\\Users\\kim ji hye2\\AppData\\Local\\anaconda3\\envs\\torch241\\Lib\\site-packages\\torch\\__init__.py'>

############ 랜덤 고정하자꾸나 ############################
SEED = 4545

import random
random.seed(SEED)       # 파이썬 랜덤 고정
np.random.seed(SEED)    # 넘파이 랜덤 고정
## 토치 시드 고정
torch.manual_seed(SEED)
## 토치 쿠다 시드 고정
torch.cuda.manual_seed(SEED)


#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
# print(x)                            # [165034 rows x 10 columns]
y = train_csv['Exited']
# print(y.shape)                      # (165034,)

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# print(np.unique(y, return_counts=True))     
# # (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))
# print(pd.DataFrame(y).value_counts())
# # 0         130113
# # 1          34921

# print(x.shape)  # (165034, 10)

x = x.to_numpy()
# x = x.reshape(165034, 5, 2)
# x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    shuffle= True,
                                                    random_state=666,
                                                    stratify=y
                                                    )

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

y_train = torch.LongTensor(y_train.values).unsqueeze(1).to(DEVICE)
y_test = torch.LongTensor(y_test.values).unsqueeze(1).to(DEVICE)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# torch.Size([132027, 10]) torch.Size([33007, 10]) torch.Size([132027, 1]) torch.Size([33007, 1])

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
    
model = Model(10, 2).to(DEVICE)    # 인풋, 아웃풋 정의
    
#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss() # 바이너리크로스 엔트로피

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train() # 훈련모드, 디폴트
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y.squeeze())  # y를 1차원으로 만들어야 함
    
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
# print("최종 loss : ", last_loss)    # 최종 loss : 

############################## 요 밑에 완성할 것 #################
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()    # 평가모드로 설정
    with torch.no_grad():
        y_predict = model(x)
        loss = criterion(y_predict, y.squeeze())
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

y_pred = torch.argmax(y_predict, dim=1).cpu().numpy()
# print(y_pred)

y_test = y_test.cpu().numpy()

acc = accuracy_score(y_test, y_pred)
print(f'accuracy: {acc:.4f}')

# 최종 loss :  0.5160576105117798
# accuracy: 0.7884