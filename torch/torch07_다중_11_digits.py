import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print('torch : ', torch.__version__, '사용DEVICE :', DEVICE)

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (1797, 64) (1797,)
# print(np.unique(y))                                     # [0 1 2 3 4 5 6 7 8 9]
# print(f"출력 뉴런 수(클래스 개수): {len(np.unique(y))}") # 출력 뉴런 수(클래스 개수): 10

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

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# torch.Size([1347, 64]) torch.Size([450, 64]) torch.Size([1347]) torch.Size([450])

#2. 모델
model = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()   # 분류 문제에 적합한 손실 함수, 스파스 크로스 엔트로피와 같다.

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
print('loss: ', loss)   # loss:  0.2834223508834839

############## acc 출력해봐욤 ##################
y_predict = model(x_test)
print(x_test[:5])
# tensor([[ 0.0000,  1.8483,  1.1983,  0.9834,  0.9791,  0.4079, -0.1041, -0.1275,
#          -0.0634, -0.0152,  0.2912, -0.7572, -0.2710,  1.2975,  2.2826, -0.1287,
#          -0.0445, -0.7423, -1.7420, -1.2221, -0.4931,  1.3174,  2.2304, -0.1131,
#          -0.0273, -0.7909, -1.4664, -1.1638,  0.8285,  1.4311, -0.6289, -0.0472,
#           0.0000, -0.6645, -1.1916, -0.9749,  0.9370,  0.8851, -0.8295,  0.0000,
#          -0.0465, -0.5254, -1.0385, -1.1292,  0.8546,  1.3616, -0.1232, -0.0925,
#          -0.0345, -0.4065, -0.6066,  0.2774,  1.2353, -0.1392, -0.7591, -0.2083,
#          -0.0273,  2.9880,  1.8370, -0.0327, -1.5897, -1.1537, -0.4973, -0.1932],
#         [ 0.0000,  0.7525,  0.5717,  0.9834,  0.9791, -0.4794, -0.4070, -0.1275,
#          -0.0634,  1.2233,  1.0293, -0.0052,  1.1958, -0.6869, -0.5151, -0.1287,
#          -0.0445, -0.4634, -0.5187, -1.2221,  1.4644, -0.6177, -0.5474, -0.1131,
#          -0.0273, -0.7909, -1.4664, -0.3143,  0.8285, -1.3010, -0.6289, -0.0472,
#           0.0000, -0.6645, -1.1916,  0.7773, -0.2419, -1.5093, -0.8295,  0.0000,
#          -0.0465, -0.5254,  0.4943,  1.0340, -1.0696, -0.7544,  0.3352, -0.0925,
#          -0.0345, -0.4065,  0.9977,  0.4663,  0.2912,  1.0196, -0.1531, -0.2083,
#          -0.0273, -0.3047,  1.2520,  0.8933,  0.0347, -0.6430, -0.4973, -0.1932],
#         [ 0.0000,  0.7525,  1.1983,  0.9834,  0.9791,  1.1178, -0.4070, -0.1275,
#          -0.0634,  1.8426,  1.0293,  0.2454,  1.1958,  1.2975,  0.3242, -0.1287,
#          -0.0445, -0.4634, -0.8682, -0.0067,  1.4644,  0.9948, -0.5474, -0.1131,
#          -0.0273, -0.7909, -1.4664,  0.3653,  0.9927, -0.6180, -0.6289, -0.0472,
#           0.0000, -0.6645, -1.1916, -1.1342,  0.7686,  0.0300, -0.8295,  0.0000,
#          -0.0465, -0.5254, -1.0385, -1.1292,  0.5339,  0.8326, -0.8107, -0.0925,
#          -0.0345, -0.4065, -0.7849, -0.4782,  1.0464,  0.8540, -0.7591, -0.2083,
#          -0.0273, -0.3047,  1.6420,  0.8933,  0.8468, -0.1322, -0.4973, -0.1932],
#         [ 0.0000, -0.3433,  0.7805,  0.7448, -1.5999, -1.0119, -0.4070, -0.1275,
#          -0.0634, -0.6344,  0.4757,  0.4961,  0.5672, -0.1908, -0.5151, -0.1287,
#          -0.0445, -0.7423, -0.6935,  1.2087,  0.4856,  0.8336, -0.2388, -0.1131,
#          -0.0273, -0.7909, -1.4664,  0.0255,  0.3360,  1.2603,  0.7427, -0.0472,
#           0.0000, -0.6645, -1.1916, -1.4528, -1.7577,  0.2010,  2.0268,  0.0000,
#          -0.0465, -0.5254, -1.0385, -1.1292, -1.2299, -0.2254,  2.3976, -0.0925,
#          -0.0345, -0.4065, -0.7849, -0.2893, -0.0865,  1.0196,  2.2708, -0.2083,
#          -0.0273, -0.3047, -0.1131, -0.0327,  0.0347,  0.3785, -0.2497, -0.1932],
#         [ 0.0000, -0.3433, -1.0992,  0.0290, -0.4277, -1.0119, -0.4070, -0.1275,
#          -0.0634, -0.6344, -1.1849,  0.9974, -1.1091, -1.3483, -0.5151, -0.1287,
#          -0.0445, -0.7423,  0.8794, -0.0067, -0.8194,  0.9948, -0.2388, -0.1131,
#          -0.0273,  1.1327,  1.1211, -1.1638, -0.1565,  1.4311,  2.3886, -0.0472,
#           0.0000,  1.9300,  1.0230, -0.0192,  0.9370,  1.0562,  0.8842,  0.0000,
#          -0.0465,  1.2299,  1.4140,  1.3430,  1.3356, -1.2834, -0.8107, -0.0925,
#          -0.0345, -0.4065, -0.9632,  0.2774,  0.6688, -1.4636, -0.7591, -0.2083,
#          -0.0273, -0.3047, -1.0881, -0.0327,  0.2377, -1.1537, -0.4973, -0.1932]])

y_predict = torch.argmax(model(x_test), dim=1)
print(y_predict[:5])
# tensor([3, 2, 3, 9, 4])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score)) # accuracy : 0.9667
print(f'accuracy : {score:.4f}')         # accuracy : 0.9667

score2 = accuracy_score(y_test.cpu().numpy(),
                        y_predict.cpu().numpy())

# print('accuracy_score : {:4.f}'.format(score2))
print(f'accuracy_score : {score2:.4f}') # accuracy_score : 0.9667

