import torch
import torchvision
from   torchvision import transforms, utils
from   torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from   PIL import Image

from   skimage import io

# In[] Read Small Testing Images from MNIST
#"""
mnist_small_train= torchvision.datasets.MNIST(
    './mnist', train=True, download=True)
#print('mnist_train:', len(mnist_train))

f=open('mnist_small_train.txt','w')
for i,(img,label) in enumerate(mnist_small_train):
    if i <= 99:
        img_path="./mnist_small_train/"+str(i)+".jpg"
        io.imsave(img_path,img)
        f.write(img_path+' '+str(label)+'\n')
    else:
        break           
f.close()
#"""

# In[] Read Training Images from MNIST
"""
mnist_test= torchvision.datasets.MNIST(
    './mnist', train=False, download=True)
print('mnist_test:', len(mnist_test))

f=open('mnist_test.txt','w')
for i,(img,label) in enumerate(mnist_test):
    #if i <= 99:
        img_path="./mnist_test/"+str(i)+".jpg"
        io.imsave(img_path,img)
        f.write(img_path+' '+str(label)+'\n')
    #else:
    #    break           
f.close()
"""

# In[] Read Testing Images from MNIST
#"""
mnist_train= torchvision.datasets.MNIST(
    './mnist', train=True, download=True)
#print('mnist_train:', len(mnist_train))

f=open('mnist_10000_train.txt','w')
for i,(img,label) in enumerate(mnist_train):
    if i <= 9999:
        img_path="./mnist_train/"+str(i)+".jpg"
        io.imsave(img_path,img)
        f.write(img_path+' '+str(label)+'\n')
    else:
        break           
f.close()
#"""

# In[] Read Testing Images from MNIST
#"""
full_mnist_train= torchvision.datasets.MNIST(
    './mnist', train=True, download=True)
#print('mnist_train:', len(mnist_train))

f=open('full_mnist_train.txt','w')
for i,(img,label) in enumerate(mnist_train):
    img_path="./mnist_train/"+str(i)+".jpg"
    io.imsave(img_path,img)
    f.write(img_path+' '+str(label)+'\n')

f.close()
#"""

# In[]
def default_loader(path):
    return Image.open(path)#.convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)




# In[]
def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0))) #.transpose((1, 2, 0))
    plt.title('Batch from dataloader')


#for i, (batch_x, batch_y) in enumerate(data_loader):

