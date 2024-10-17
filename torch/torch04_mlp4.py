import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE', DEVICE)

# 1. 데이터 (차원 유지)
x = np.array([range(10)]).transpose()  # (10, 1)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]).transpose()  # (10, 3)

# Numpy 배열을 Tensor로 변환
x = torch.Tensor(x).to(DEVICE)  # (10, 1)
y = torch.Tensor(y).to(DEVICE)  # (10, 3)

# 2. 모델
model = nn.Sequential(
    nn.Linear(1, 10),  # 입력 크기를 1로 설정
    nn.Linear(10, 9),
    nn.Linear(9, 8),
    nn.Linear(8, 7),
    nn.Linear(7, 6),
    nn.Linear(6, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),  # 출력이 3개가 되도록 설정
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    model.train()  # 모델을 학습 모드로 설정
    optimizer.zero_grad()
    
    hypothesis = model(x)  # (10, 1) -> (10, 3)
    loss = criterion(hypothesis, y)  # 손실 계산
    
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 업데이트
    
    return loss.item()

# 4. 학습 루프
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

# 5. 평가 함수 정의
def evaluate(model, criterion, x, y):
    model.eval()  # 모델을 평가 모드로 설정
    
    with torch.no_grad():
        y_predict = model(x)  # (10, 1)
        loss2 = criterion(y_predict, y)
    return loss2.item()

# 6. 평가 과정
loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

# 7. 예측
# [10]을 입력으로 넣어 3개의 예측값을 구함
# result_input = torch.Tensor([[10]]).to(DEVICE)  # 입력은 1차원이어야 함
# results = model(result_input)  # 예측 수행
# print('[10] 의 예측값 : ', results.cpu().detach().numpy())

# 예측값 : [10]이라고 선생님이 말씀하심.

# 최종 loss :  0.005317175295203924
# [10] 의 예측값 :  [[ 1.1000001e+01  1.5733339e+00 -5.9604645e-08]]


