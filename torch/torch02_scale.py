# torch01_2_gpu.py 복사

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# GPU
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)
# torch :  2.4.1+cu124 사용DEVICE :  cuda

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x2 = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) # (3,) -> (3,1) GPU로 돌리겠다!!!
print('스케일링 전 : ', x)  # 스케일링 전 :  [1 2 3]

x = (x2 - torch.mean(x2)) / torch.std(x2)
print('스케일링 후 : ', x)  
# 스케일링 후 :  tensor([[-1.],
        # [ 0.],
        # [ 1.]], device='cuda:0') 

y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   # (3,) -> (3,1)
print(x.shape, y.shape)   # torch.Size([3, 1]) torch.Size([3, 1])
print(x.size(), y.size()) # torch.Size([3, 1]) torch.Size([3, 1])

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Linear(1, 1).to(DEVICE) # 인풋, 아웃풋  # y = xw + b

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

x_pre = (torch.Tensor([[4]]).to(DEVICE) - torch.mean(x2)) / torch.std(x2)
print(x_pre)

result = model(x_pre)     # model의 가중치가 최상의 가중치가 저장되어 있어 tensor형태의 값으로 넣어줌

print("4의 예측값 :", result)              # 4의 예측값 : tensor([[3.9947]], grad_fn=<AddmmBackward0>)      # item 미사용시 gradient도 나옴 
print("4의 예측값 :", result.item()) 

# 최종 loss :  1.1227760041143675e-11
# tensor([[2.]], device='cuda:0')
# 4의 예측값 : tensor([[4.0000]], device='cuda:0', grad_fn=<AddmmBackward0>)
# 4의 예측값 : 3.999992847442627