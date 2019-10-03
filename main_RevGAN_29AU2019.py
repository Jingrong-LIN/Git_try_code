# In[]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from   torch.utils.data import Dataset, DataLoader
from   torch.autograd import Variable

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
from   skimage import io
import time
from IPython.display import HTML

from dataset_cGAN  import MyDataset
from fdataset_cGAN import fImage 
from net_RevGAN_29AU2019   import imageGenerator, discriminator
from utils_cGAN    import conv3x3, conv3x3T, CEmisfit, getAccuracy
import config_RevGAN_29AU2019, dataset_cGAN, fdataset_cGAN

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#print(device)
#device = "cpu"
cost_func = nn.CrossEntropyLoss()


# In[] Read Data
train_small_data=MyDataset(txt='mnist_small_train.txt', transform=transforms.ToTensor())
Train_Small_dataloader = DataLoader(train_small_data, batch_size=bs, shuffle=True, drop_last=True)
print('Length of Small Train Loader:', len(Train_Small_dataloader))

train_full_data=MyDataset(txt='mnist_full_train.txt', transform=transforms.ToTensor())
Train_Full_dataloader = DataLoader(train_full_data, batch_size=bs, shuffle=True, drop_last=True)
print('Length of Full Train Loader:', len(Train_Full_dataloader))

train_10000_data=MyDataset(txt='mnist_10000_train.txt', transform=transforms.ToTensor())
Train_10000_dataloader = DataLoader(train_10000_data, batch_size=bs, shuffle=True, drop_last=True)
print('Length of 10000 Train Loader:', len(Train_10000_dataloader))

test_data=MyDataset(txt='mnist_test.txt', transform=transforms.ToTensor())
Test_dataloader  = DataLoader(test_data, batch_size=bs, shuffle=True, drop_last=True)
print('Length of Test Loader:', len(Test_dataloader))


# In[]
optimizerGen = optim.SGD([{'params':KGEN}], lr=1e-3, momentum=0)
optimizerDis = optim.SGD([{'params':KY},{'params':KZ},{'params':WY},{'params':WZ}], lr=1e-3, momentum=0)


ones_label  = Variable(torch.ones(bs, dtype=torch.long)).to(device) #  .long() for CE
zeros_label = Variable(torch.zeros(bs, dtype=torch.long)).to(device)#  .long()

netGen    = imageGenerator(h_G,NG_G)
netDisY   = discriminator(h,NG)
netDisZ   = discriminator(h,NG)


# In[]
start = time.time()
for epoch in range(1):  # loop over the dataset multiple times

    running_accuracyZZ = 0.0
    running_accuracyZY = 0.0
    running_accuracyYZ = 0.0
    running_accuracyYY = 0.0
    
    running_lossZZ  = 0.0
    running_lossZY  = 0.0
    running_lossYZ  = 0.0
    running_lossYY  = 0.0

    Input_dataloader = Train_10000_dataloader ; 

    for i, Rmnist in enumerate(Input_dataloader):

        Y,_    = Rmnist
        Y      = Y.to(device)
        Yreal  = torch.cat((Y,Y,Y),dim=1)
                
        Rtmp, _ = next(iter(Input_dataloader)); Rtmp = Rtmp.to(device)
        Zreal  = fImage(Rtmp, Kf)#.to(device)
        #Zreal   = torch.cat((Z,Z,Z),dim=1)
        
        Zfake   = netGen.forward(Yreal,  KGEN)
        Yfake   = netGen.backward(Zreal, KGEN)
                 
        # run the network
        pYreal = netDisY(Yreal,  KY)
        pYfake = netDisY(Yfake,  KY)
        pZreal = netDisZ(Zreal,  KZ)
        pZfake = netDisZ(Zfake,  KZ)
        
        lossYZ, SYZi = CEmisfit(pZfake,WZ,zeros_label)
        lossZZ, SZZi = CEmisfit(pZreal,WZ, ones_label)
        lossZY, SZYi = CEmisfit(pYfake,WY,zeros_label)
        lossYY, SYYi = CEmisfit(pYreal,WY, ones_label)
        
        optimizerGen.zero_grad()
        lossG = torch.log(1.0-lossYZ) + torch.log(1-lossZY)
        lossG.backward(retain_graph=True)
        optimizerGen.step()
        
        ###
        optimizerDis.zero_grad()
        lossD = -(torch.log(lossYY) + torch.log(lossZZ) + torch.log(1.0-lossYZ) + torch.log(1-lossZY))
        lossD.backward()
        optimizerDis.step()    
        
        # print statistics
        accuracyZY = getAccuracy(SZYi,zeros_label)
        accuracyZZ = getAccuracy(SZZi, ones_label)
        accuracyYZ = getAccuracy(SYZi,zeros_label)
        accuracyYY = getAccuracy(SYYi, ones_label)
        
        running_lossYY  += lossYY.item()
        running_lossZZ  += lossZZ.item()
        running_lossYZ  += lossYZ.item()
        running_lossZY  += lossZY.item()
        
        running_accuracyYY += accuracyYY
        running_accuracyZZ += accuracyZZ
        running_accuracyYZ += accuracyYZ
        running_accuracyZY += accuracyZY
        
        n_print = 200
        if i % n_print == n_print-1:    # print every 2000 mini-batches
            print('[%d/%d, %d/%d]  %.3f  %.3f  %.3f  %.3f' %
                 (epoch + 1, Epoch, i + 1, len(Input_dataloader),
                                   running_accuracyYY/n_print, running_accuracyZZ/n_print, 
                                   running_accuracyZY/n_print, running_accuracyYZ/n_print))
            print(torch.cuda.memory_allocated()*1e-6, torch.cuda.memory_cached()*1e-6)
             
            print( '\t lossD: %.4f,  \t  lossG: %.4f '
                  % (lossD, lossG))
            
            running_accuracyZZ = 0.0
            running_accuracyYY = 0.0
            running_accuracyYZ = 0.0
            running_accuracyZY = 0.0

print('Finished Training')
end = time.time()
# In[]

    
    Yv, _       = next(iter(Test_dataloader)) ; Yv = Yv.to(device)
    Yvreal      = torch.cat((Yv,Yv,Yv),dim=1)
    Ytmpv, _    = next(iter(Test_dataloader)) ; Ytmpv = Ytmpv.to(device)
    Ytmpv       = torch.cat((Ytmpv,Ytmpv,Ytmpv),dim=1)
    Zvreal      = fImage(Ytmpv,Kf)
    Zvfake      = netGen(Yvreal,K0f,B0f,KL,BL,h); 
    Yvfake      = netGen.backward(Zvreal,K0b,B0b,KL,BL,h)
  




    
