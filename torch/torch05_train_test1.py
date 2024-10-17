import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  2.4.1+cu124 사용 DEVICE :  cuda

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7]).transpose()
y_train = np.array([1,2,3,4,5,6,7]).transpose()
x_test = np.array([8,9,10,11]).transpose()
y_test = np.array([8,9,10,11]).transpose()
x_predict = np.array([12, 13, 14]).transpose()

###############################
# [실습] 맹그러봐!!!!
###############################

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
x_predict = torch.FloatTensor(x_predict).unsqueeze(1).to(DEVICE)

#2. 모델
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 1) 
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    
    hyporthesis = model(x)
    loss = criterion(hyporthesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 100
for epoch in range(1, epochs +1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 10 == 0:
        print(f"epoch : {epoch}, loss : {loss:.6f}")
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
        return loss2.item()
    
loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss ", loss2)

result = model(x_test.to(DEVICE))
print(result.cpu().detach().numpy())

print('=============================')

# 예측할 데이터 [12, 13, 14]에 대한 예측값 출력
result = model(x_predict.to(DEVICE))
print("[12, 13, 14]의 예측 결과: ", result.cpu().detach().numpy())

# 최종 loss  1.8996852304553613e-05
# [[ 8.005432]
# [ 9.004658]
# [10.003885]
# [11.003113]]
# =============================
# [12, 13, 14]의 예측 결과:  [[12.002339]
# [13.001566]
# [14.000794]]
