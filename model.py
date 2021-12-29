# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:58:59 2019

@author: ricky
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from qpth.qp import QPFunction, QPSolvers
import time


class Net(nn.Module):
    
    def __init__(self, nHidden=128,nCls=10):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,16,5)
        self.conv2 = nn.Conv2d(16,32,5)
#        self.pool = nn.MaxPool2d(2,1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(32*4*4,nHidden)
        self.fc2 = nn.Linear(nHidden,nCls)
        
        self.Q = Variable(0.5*torch.eye(nCls).double().cuda())
        self.G = Variable(-torch.eye(nCls).double().cuda())
        self.h = Variable(-1e-5*torch.ones(nCls).double().cuda())
        self.A = Variable((torch.ones(1, nCls)).double().cuda())
        self.b = Variable(torch.Tensor([1.]).double().cuda())
        def projF(x):
                nBatch = x.size(0)
                Q = self.Q.unsqueeze(0).expand(nBatch, nCls, nCls)
                G = self.G.unsqueeze(0).expand(nBatch, nCls, nCls)
                h = self.h.unsqueeze(0).expand(nBatch, nCls)
                A = self.A.unsqueeze(0).expand(nBatch, 1, nCls)
                b = self.b.unsqueeze(0).expand(nBatch, 1)
                x = QPFunction()(Q, -x.double(), G, h, A, b).float()
                x = x.log()
                return x
        self.projF = projF

        
    def forward(self,data):
        data = F.relu(self.conv1(data)) # (24,24,16)
        data = F.max_pool2d(data,2) # (12,12,16)
#        data = self.pool(data)
        data = F.relu(self.conv2(data)) # (8,8,32)    
#        data = self.pool(data)
        data = F.max_pool2d(data,2) # (4,4,32)
#        data = data.view(-1,5*5*32)
#        data = self.pool(data)
        # data = self.dropout1(data)
        data = torch.flatten(data,1)
        data = F.relu(self.fc1(data))
        # data = self.dropout2(data)
        data = self.fc2(data)
        return self.projF(data)

#        
#        
def train(model,device,train_loader,optimizer):
    model.train()
    running_loss = 0.0
    for count, data in enumerate(train_loader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('\n Train Loss:{:.6f}'.format(running_loss/count))
    return running_loss/count
        
        
def test(model,device,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for testimg,labels in test_loader:
            testimg, labels = testimg.to(device), labels.to(device)
            output = model(testimg)
            test_loss += F.cross_entropy(output, labels, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\n Test loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss
    
#def imshow(img):
#    img = img/2 + 0.5
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg,(1,2,0)))
#    plt.show()

    
    
def main():
    start_time = time.time()
    batch_size = 64
    num_epoch = 20
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
#    print(torch.cuda.is_available())
#    print(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
        
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=batch_size, shuffle=False, **kwargs)

    
    model = Net().to(device)
#    criterion = F.cross_entropy()

#    optimizer = optim.SGD(model.parameters(),lr=0.005,momentum=0.9,nesterov=True)  
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    train_loss = []
    test_loss = []
    
    trainF = open('adadelta_train.csv', 'w')
    testF = open('adadelta_test.csv', 'w')
    
    for epoch in range(1,num_epoch+1):
        print('{}-th'.format(epoch))
        train_loss.append(train(model,device,train_loader,optimizer))
        test_loss.append(test(model,device,test_loader))
        scheduler.step()
        
        
        
    print(train_loss)
    print(test_loss)
    np.savetxt(trainF, train_loss)
    np.savetxt(testF,test_loss)
    
    trainF.close()
        
    testF.close()
        
#     idx = range(1,num_epoch+1)    
#     plt.figure(figsize=(5,5))
#     plt.plot(idx, train_loss, "*-",color='blue')
#     plt.plot(idx, test_loss, "*-",color='olive')
# #    plt.yticks(np.arange(0, 0.35, step=0.05))
# #    plt.xticks(np.arange(0, 24, step=4))
# #    plt.ylim(2.1, 2.4)
#     plt.xticks([4, 8, 12, 16, 20])
#     plt.legend(['train_loss', 'test_loss'], loc='upper right')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.show()
        
    torch.save(model.state_dict(), "mnist_cnn.pt")
    end_time = time.time()
    elasped_time = end_time-start_time
    print(elasped_time)
    

if __name__ == '__main__':
    main()    
    
    
    