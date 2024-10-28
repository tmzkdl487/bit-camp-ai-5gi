import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
# print('torch :', torch.__version__, '사용자DEVICE: ', DEVICE)

DEVICE = 'cuda:0' if torch.cuda.is_available else 'cpu'
print(DEVICE)

path = 'C:/ai5/_data/kaggle/netflix/'
train_csv = pd.read_csv(path + 'train.csv')
print(train_csv)    # [967 rows x 6 columns]
print(train_csv.info()) 
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 967 entries, 0 to 966
# Data columns (total 6 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   Date    967 non-null    object
#  1   Open    967 non-null    int64
#  2   High    967 non-null    int64
#  3   Low     967 non-null    int64
#  4   Volume  967 non-null    int64
#  5   Close   967 non-null    int64
# dtypes: int64(5), object(1)
# memory usage: 45.5+ KB
# None

print(train_csv.describe())
#              Open        High         Low        Volume       Close
# count  967.000000  967.000000  967.000000  9.670000e+02  967.000000
# mean   223.923475  227.154085  220.323681  9.886233e+06  223.827301
# std    104.455030  106.028484  102.549658  6.467710e+06  104.319356
# min     81.000000   85.000000   80.000000  1.616300e+06   83.000000
# 25%    124.000000  126.000000  123.000000  5.638150e+06  124.000000
# 50%    194.000000  196.000000  192.000000  8.063300e+06  194.000000
# 75%    329.000000  332.000000  323.000000  1.198440e+07  327.500000
# max    421.000000  423.000000  413.000000  5.841040e+07  419.000000

# import matplotlib.pyplot as plt
# # data = train_csv[1:4]   # 에러남
# data = train_csv.iloc[:, 1:4]
# data['종가'] = train_csv['Close']
# print(data)

# hist = data.hist()
# plt.show()

###### 요런 문제 있어, 뭔 문제? 컬럼별로 최소, 회대가 아닌 전체 데이터셋의 최대 최소!!!
# data = train_csv.iloc[:, 1:4].values
# data = (data - np.min(data)) / (np.max(data) - np.min(data))
# data = pd.DataFrame(data)
# print(data.describe())

############### 요렇게 axis=0을 넣어주면 컬럼별로 최대, 최소를 구한다. 
# data = train_csv.iloc[:, 1:4].values
# data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
# data = pd.DataFrame(data)
# print(data.describe())

from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self):
        self.csv = train_csv
        
        self.x = self.csv.iloc[:, 1:4].values   # 시가, 고가, 저가 컬럼이 됨.
        self.x = (self.x - np.min(self.x, axis=0))/ (np.max(self.x, axis=0) - np.min(self.x, axis=0)) # 정규화
        
        self.y = self.csv['Close'].values
    
    def __len__(self):
        return len(self.x) - 30
    
    def __getitem__(self, i):
        x = self.x[i:i+30]
        y = self.y[i+30]

        return x, y

aaa = Custom_Dataset()
# print(aaa)          # <__main__.Custom_Dataset object at 0x000002573BC10830>
# print(type(aaa))    # <class '__main__.Custom_Dataset'>
# print(aaa[0])       
# (array([[0.11470588, 0.11242604, 0.11411411],
#        [0.12647059, 0.12130178, 0.12612613],
#        [0.11764706, 0.10946746, 0.11411411],
#        [0.11470588, 0.1035503 , 0.10810811],
#        [0.10588235, 0.09467456, 0.10510511],
#        [0.10588235, 0.10059172, 0.10810811],
#        [0.10882353, 0.10059172, 0.11111111],
#        [0.10588235, 0.09467456, 0.1021021 ],
#        [0.10882353, 0.1035503 , 0.11111111],
#        [0.11176471, 0.10059172, 0.10810811],
#        [0.10294118, 0.09467456, 0.1021021 ],
#        [0.08235294, 0.0739645 , 0.07507508],
#        [0.08529412, 0.07692308, 0.07807808],
#        [0.07058824, 0.09763314, 0.07507508],
#        [0.10294118, 0.10946746, 0.0960961 ],
#        [0.10294118, 0.09763314, 0.09309309],
#        [0.09117647, 0.09467456, 0.09309309],
#        [0.10294118, 0.09763314, 0.10510511],
#        [0.09705882, 0.08579882, 0.07507508],
#        [0.07352941, 0.07100592, 0.06306306],
#        [0.06176471, 0.06213018, 0.06606607],
#        [0.07647059, 0.0739645 , 0.07807808],
#        [0.08235294, 0.0739645 , 0.05105105],
#        [0.07941176, 0.07100592, 0.06606607],
#        [0.07058824, 0.0591716 , 0.05705706],
#        [0.05588235, 0.05325444, 0.05705706],
#        [0.05588235, 0.04733728, 0.04504505],
#        [0.04705882, 0.03846154, 0.03303303],
#        [0.03823529, 0.0295858 , 0.03003003],
#        [0.03235294, 0.02662722, 0.03303303]]), 94)
# 세어보면 30개. 잘 나왔다.

# print(aaa[0][0].shape)  # (30, 3)
# print(aaa[0][1])        # 94
# print(len(aaa))         # 937
# print(aaa[937])         # 936번째까지 있지. 936번째 놈이 끝! IndexError: index 967 is out of bounds for axis 0 with size 967

##### x는 (937, 30, 3), y는 (937, 1)

train_loader = DataLoader(aaa, batch_size= 32)

# aaa = iter(train_loader)
# bbb = next(aaa)
# print(bbb)
# print(bbb[0].size())    # torch.Size([32, 30, 3])

#2. 모델
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size=3, 
                          hidden_size=64, 
                          num_layers= 5,
                          batch_first=True,                          
                          )
        self.fc1 = nn.Linear(in_features=30*64, out_features=32)
        self.fc2 = nn.Linear(in_features=32,  out_features=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)
        
        x = torch.reshape(x, (x.shape[0], -1))
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
model = RNN().to(DEVICE)

#3. 컴파일, 훈련

from torch.optim import Adam
optim = Adam(params=model.parameters(), lr=0.001)

import tqdm 

for epoch in range(1, 201):
    iterator = tqdm.tqdm(train_loader)
    for x, y in iterator:
        optim.zero_grad()
        
        h0 = torch.zeros(5, x.shape[0], 64).to(DEVICE) # (num_layers, batch_size, hidden_size) = (5, 32, 64)
        
        hypothesis = model(x.type(torch.FloatTensor).to(DEVICE), h0)
        
        loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))

        loss.backward()
        optim.step()
        
        iterator.set_description(f'epoch:{epoch} loss:{loss.item()}')

save_path = 'C:/ai5/_save/torch/'
torch.save(model.state_dict(), save_path + 't22.pth')

        