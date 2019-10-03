# In[]
import torch
import torch.nn as nn
import torch.nn.functional as F



# In[]
def conv3x3(x,K):
    """3x3 convolution with padding"""
    return F.conv2d(x, K, stride=1, padding=1)

def conv3x3T(x,K):
    """3x3 convolution transpose with padding"""
    #K = torch.transpose(K,0,1)
    return F.conv_transpose2d(x, K, stride=1, padding=1)

def conv1x1(x,K):
    """1x1 convolution"""
    return F.conv2d(x, K, stride=1, padding=0)

def convDiag(x,K):
    n = K.shape
    return F.conv2d(x, K, stride=1, padding=1, groups=n[0])

def coarseImg(x,res):
    n = x.shape[2]
    while n>res:
        x = F.avg_pool2d(x, 2, 2, 0, False,False)
        n = x.shape[2]        
    return x

def tvNorm(T):
    t = torch.sum(T**2,1).unsqueeze(1)/T.shape[1]
    return T/torch.sqrt(t+1e-3)

dis = nn.CrossEntropyLoss()

def CEmisfit(X,W,C):
    n = W.shape
    X = X.view(-1,n[0])
    S = torch.matmul(X,W)
    return dis(S,C), S   

def getAccuracy(S,C):
    values, indices = torch.max(S.data, 1)
    batchsize = C.size(0)
    correct = (indices == C).sum().item()
    return correct/batchsize

def L2misfit(X,W,C):
    n = W.shape
    X = X.view(-1,n[0])
    S = torch.matmul(X,W)
    return 0.5*torch.norm(S-C)**2   
