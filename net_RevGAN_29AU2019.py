# In[]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from   utils_cGAN       import conv3x3, conv3x3T, tvNorm

# In[]
class imageGenerator(nn.Module):

    def __init__(self, h,NG):
        super().__init__()

        # network geometry
        self.NG       = NG
        # time step
        self.h        = h
        
    def forward(self,x,Kresnet):
    
        nt = len(Kresnet)      
        # first step: use Forwad Euler
        xold = x
        z1  = conv3x3(x, Kresnet[0])
        z2  = conv3x3T(x, Kresnet[0])
        
        z  = tvNorm(z1-z2)
        z  = F.relu(z)        
        x  = x + self.h*z   
            
        # time stepping
        for j in range(1,nt-1):
            
            z1  = conv3x3(x, Kresnet[j])
            z2  = conv3x3T(x, Kresnet[j])
        
            z  = tvNorm(z1-z2)
            z  = F.relu(z)
            tmp = x
            # Leapfrog test
            x  = xold + 2*self.h*z   

            xold = tmp
        return x

    def backward(self,x,Kresnet,xold=[]):
    
        xold = x
        
        nt = len(Kresnet)   
        # first step
        z1  = conv3x3(x, Kresnet[-1])
        z2  = conv3x3T(x, Kresnet[-1])
        
        z  = tvNorm(z1-z2)
        z  = F.relu(z)  
        # First step Forward Euler
        x  = x - self.h*z   

        # time stepping backward
        for j in np.flip(range(1,nt-1)):
            
            z1  = conv3x3(x, Kresnet[j])
            z2  = conv3x3T(x, Kresnet[j])
        
            z  = tvNorm(z1-z2)
            z  = F.relu(z)
            tmp = x
            # leapfrog step
            x  = xold - 2*self.h*z   
            xold = tmp
        return x
    
    
# In[]
class discriminator(nn.Module):

    def __init__(self, h,NG):
        super().__init__()

        # network geometry
        self.NG       = NG
        # time step
        self.h        = h
        
    def forward(self,x,Kresnet):
    
        nt = len(Kresnet)        
        # time stepping
        for j in range(nt):
            
            # First case - rsent style step
            if self.NG[0,j] == self.NG[2,j]: 
                z  = conv3x3(x, Kresnet[j])
                z  = F.instance_norm(z)
                z  = F.relu(z)        
                z  = conv3x3T(z,Kresnet[j])
                x  = x - self.h*z
            # Change number of channels/resolution    
            else:
                z  = conv3x3(x, Kresnet[j])
                z  = F.instance_norm(z)
                x  = F.relu(z)
                if self.NG[3,j] == 1:
                    x = F.avg_pool2d(x,2)
                         
        return x        