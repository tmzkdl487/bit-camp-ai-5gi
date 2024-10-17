import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE', DEVICE)

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)]).transpose()

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [10, 9,8,7,6,5,4,3,2,1]
              ]).transpose()
# print(x.shape, y.shape) # (3, 10) (10, 3)

## 맹그러봐!!! 
x = torch.Tensor(x).to(DEVICE) # (3, 10)
y = torch.Tensor(y).to(DEVICE) # (10, 3)

#2. 모델
model = nn.Sequential(
    nn.Linear(3, 10),
    nn.Linear(10, 9),
    nn.Linear(9, 8),
    nn.Linear(8, 7),
    nn.Linear(7, 6),
    nn.Linear(6, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3), # 마지막 출력이 3개가 되도록 설정
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    model.train()  # 모델을 학습 모드로 설정
    optimizer.zero_grad()
    
    hypothesis = model(x) # (10, 3)
    loss = criterion(hypothesis, y) # 손실 계산
    
    loss.backward()  # 역전파
    optimizer.step() # 가중치 업데이트
    
    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    if epoch % 100 == 0:
      print(f'epoch: {epoch}, loss: {loss}')
        
    # NaN 값 발생 시 학습 중단
    if torch.isnan(torch.tensor(loss)):
        print("NaN 발생, 학습을 중단합니다.")
        break
    
print("=============================================")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()  # 모델을 평가 모드로 설정
    
    with torch.no_grad():
        y_predict = model(x) # (10, 3)
        loss2 = criterion(y_predict, y)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

result_input = model(torch.Tensor([[10, 31, 211]]).to(DEVICE))  # 예측하려는 입력도 Tensor로 변환
results = model(result_input) 
print('[10, 31, 211] 의 예측값 : ', results.cpu().detach().numpy()) 

# 예측값 : [10, 31, 211]
# 최종 loss :  0.0053171683102846146
# [10, 31, 211] 의 예측값 :  [[ 6.314491    0.17934707 -6.440175  ]]

# ================================================= 선생님이 알려주신 방법
result = model(torch.Tensor([[10, 1.3, 1]]).to(DEVICE))
print(result) # tensor([[ 6.8830,  0.6136, -6.7127]], device='cuda:0',
              # grad_fn=<AddmmBackward0>)
print(result.detach())  # tensor([[8.1015, -0.0430, -7.9610]], device-'cuda:0')
# Print(result.detach().numpy()) # numpy는 cpu에서만 돌아서 에러남
# print(result.detach().cpu().numpy())

print('[10, 1.3, 1]의 예측값 : ', result.detach().cpu().numpy())