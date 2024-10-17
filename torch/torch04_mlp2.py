import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE', DEVICE)

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
              ]).transpose()

y = np.array([1,2,3,4,5,6,7,7,9,10])
# print(x.shape, y.shape) # (10, 3) (10,)

# 맹그러봐!

x = torch.Tensor(x).to(DEVICE)
y = torch.Tensor(y).view(-1, 1).to(DEVICE)

#2. 모델
model = nn.Sequential(
    nn.Linear(3, 10),
    nn.Linear(10, 9),
    nn.Linear(9, 8),
    nn.Linear(8, 7),
    nn.Linear(7, 6),
    nn.Linear(6, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    model.train()  # 모델을 학습 모드로 설정
    optimizer.zero_grad()
    hypothesis = model(x) # 예측값
    loss = criterion(hypothesis, y) # 손실 계산
    
    loss.backward()  # 역전파
    optimizer.step() # 가중치 업데이트
    
    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss: {loss}')
    
print("=============================================")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()  # 모델을 평가 모드로 설정
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

results = model(torch.Tensor([[10, 1.3, 1]]).to(DEVICE))  # 예측하려는 입력도 Tensor로 변환
print('[10] 의 예측값 : ', results.detach().cpu().numpy()) 

# 예측값 : [10, 1.3, 1] 이라고 선생님이 알려주심.

# 3개 구해야되는 줄 알고 잘못 코드 짜서 3개 나옴.
# 최종 loss :  7.840135097503662
# [10, 1.3, 1] 의 예측값 :  [[ 5.4211993  -0.66548455  0.37630957]]

# 다시 코드 잘 짜서 10이 나오게 변경함.
# 최종 loss :  0.08358428627252579
# [10] 의 예측값 :  [[9.890343]]