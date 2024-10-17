import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# GPU
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# x = torch.FloatTensor(x)
# print(x.shape)  # torch.Size([3])
# print(x.size()) # torch.Size([3])라고 나옴. .shape로도 똑같이 나옴. 토치는 보통 size로 씀

# 2차원 행렬 형태로 만들어줘야함.
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) # (3,) -> (3,1) GPU로 돌리겠다!!!
# print(x)
# tensor([[1.],
#        [2.],
#        [3.]])

y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   # (3,) -> (3,1)
print(x.shape, y.shape)   # torch.Size([3, 1]) torch.Size([3, 1])
print(x.size(), y.size()) # torch.Size([3, 1]) torch.Size([3, 1])

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model = nn.Linear(1, 1).to(DEVICE) # 인풋, 아웃풋  # y = xw + b
model = nn.Sequential(
    nn.Linear(1, 5), 
    nn.Linear(5, 4), 
    nn.Linear(4, 3), 
    nn.Linear(3, 2), 
    nn.Linear(2, 1), 
).to(DEVICE)

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y): 
    # model.train()         # 훈련모드
    optimizer.zero_grad()   # 각 배치마다 기울기를 초기화하여, 기울기 누적에 의한 문제 해결, 반드시 zero_grad할것.
    
    hypothesis = model(x)   # y = wx + b
    
    loss = criterion(hypothesis, y) # loss = mse()
    
    loss.backward()     # 기울기(gradient)값 계산까지. # 역전파 시작  # 로스는 웨이트로 미분
    optimizer.step()    # 가중치(w) 갱신               # 역전파 끝

    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}' .format(epoch, loss))   # verbose 

print("==========================================================")    
    
# 4. 평가, 예측
# loss = model.evaluate(x, y)

def evaluate(model, criterion, x, y):
    model.eval()    # 평가모드 / 역전파X, 가중치X, 기울기 갱신X
    
    with torch.no_grad():       # 이거 안쓰면 기울기 누적될 수 있으니 반드시 쓸것.
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

results = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값 : :', results.item())    

# 데이터, 모델 / 투 토치. 디바이스

# 최종 loss :  1.7053025658242404e-12
# 4의 예측값 : : 3.999997138977051

