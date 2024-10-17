import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split  # train_test_split 임포트

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

# 1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

# train_test_split을 사용하여 데이터를 훈련과 테스트 세트로 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Numpy 배열을 Tensor로 변환 및 차원 맞추기 (N, 1)
x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# 2. 모델
model = nn.Sequential(
    nn.Linear(1, 5),  # 입력 차원이 1
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 1)   # 최종 출력 차원 1로 설정
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.MSELoss()  # 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.1)  # 옵티마이저 설정

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()  # 그래디언트 초기화
    
    hypothesis = model(x)  # 예측값
    loss = criterion(hypothesis, y)  # 손실 계산
    
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 업데이트
    
    return loss.item()

# 훈련 루프
epochs = 100
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 10 == 0:
        print(f"epoch : {epoch}, loss : {loss:.6f}")
    
# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()  # 평가 모드로 전환
    
    with torch.no_grad():
        prediction = model(x)
        loss = criterion(prediction, y)
        return loss.item()

# 평가
loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss: ", loss2)

# 테스트 데이터에 대한 예측 결과
result = model(x_test.to(DEVICE))
print("테스트 데이터 예측 결과: ", result.cpu().detach().numpy())

# 최종 loss:  0.05299583077430725
# 테스트 데이터 예측 결과:  [[84.17779  ]
#  [54.216373 ]
#  [71.1945   ]
#  [46.22666  ]
#  [45.227955 ]
#  [40.234386 ]
#  [23.256254 ]
#  [81.18163  ]
#  [11.271695 ]
#  [ 1.2845609]
#  [19.261402 ]
#  [31.245964 ]
#  [74.19064  ]
#  [34.242104 ]
#  [91.16876  ]
#  [ 5.2794147]
#  [77.18679  ]
#  [78.18549  ]
#  [13.269123 ]
#  [32.244675 ]]