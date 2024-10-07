import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE', DEVICE)

#1. 데이터
x = np.array([range(10)])

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [10, 9,8,7,6,5,4,3,2,1]
              ]).transpose()
print(x.shape, y.shape) # 

# 예측값 : [10]