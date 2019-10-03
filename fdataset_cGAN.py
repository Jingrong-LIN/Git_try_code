# In[]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from   torchvision import datasets, transforms
from   PIL import Image, ImageFilter
import matplotlib.pyplot as plt

from   utils_cGAN  import conv3x3



# In[]  

def fImage(X, Kf):

    v = []
                
    v00 = 5*X/torch.max(X)
    v01 = 5*X/torch.max(X)
    v02 = 5*X/torch.max(X)
    v.append(v00); v.append(v01); v.append(v02)  
    
    for i in range(20):
        for j in range(2):        
            z = conv3x3(v[j], Kf[j])
            z = torch.log(torch.abs(z) + 1E-3)
            v[j] = -v[j] + 0.05*(conv3x3(z,Kf[j]))
    fmnist = torch.cat((v[0], v[1], v[2]), dim = 1)

    return fmnist

