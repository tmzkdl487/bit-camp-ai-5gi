import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print('torch : ', torch.__version__, '사용DEVICE :', DEVICE)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (178, 13) (178,)

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
    nn.Linear(13, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 13)
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
print('loss: ', loss)   # loss:  0.313359797000885

############## acc 출력해봐욤 ##################
y_predict = model(x_test)
print(x_test[:5])
# tensor([[-1.9194,  1.2376, -1.8932,  0.0367,  0.4783,  1.4423,  0.6097, -0.9939,
#           3.6549, -0.8829, -0.8983,  0.2528, -0.5968],
#         [ 0.6866, -0.6237, -0.4054,  1.4163, -0.8511, -0.6777, -0.1617, -0.7491,
#          -0.9693, -0.5239,  0.0765,  0.2096, -0.8933],
#         [-1.2262,  0.9602, -1.2555, -0.1166, -0.8511, -0.4820, -0.3702,  0.0669,
#           0.5540, -1.5753, -0.1354,  0.5984, -0.5902],
#         [-0.6742, -0.7401, -0.2283,  0.6498, -0.9176,  0.7248,  1.2038,  0.2301,
#           0.3726, -0.4384, -1.1526,  0.2960, -1.2887],
#         [-1.7269, -0.9011,  1.2241,  0.1899, -0.3858,  0.7248,  0.9641, -0.5859,
#           1.6783, -0.9897, -0.0083,  0.9008, -0.2080]])

y_predict = torch.argmax(model(x_test), dim=1)
print(y_predict[:5])
# tensor([1, 1, 1, 1, 1])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score)) # accuracy : 0.9778
print(f'accuracy : {score:.4f}')         # accuracy : 0.9778

score2 = accuracy_score(y_test.cpu().numpy(),
                        y_predict.cpu().numpy())

# print('accuracy_score : {:4.f}'.format(score2))
print(f'accuracy_score : {score2:.4f}') # accuracy_score : 0.9778
