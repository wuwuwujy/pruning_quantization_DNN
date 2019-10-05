import math
import numpy as np
import time
import torch
import torch.nn.init as init
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
from torchvision.transforms import transforms
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layerblock1 = nn.Sequential(
            nn.Conv2d(3  , 64 , kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 , 64 , kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)

        )
        self.layerblock2 = nn.Sequential(
            nn.Conv2d(64 , 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)

        )
        self.layerblock3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.layerblock4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.layerblock5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.layerblock1(x)
        x = self.layerblock2(x)
        x = self.layerblock3(x)
        x = self.layerblock4(x)
        x = self.layerblock5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":

    filepath = '../CIFAR10_dataset'
    #data loading and augmentation
    #cifar 10 normalization parameter https://arxiv.org/pdf/1909.12205.pdf
    train_transformations = transforms.Compose([transforms.RandomHorizontalFlip(), 
                        transforms.RandomCrop(32, padding = 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])
    test_transformations = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])
    train_dataset = CIFAR10(root = filepath, train=True, download=True, transform = train_transformations)
    test_dataset = CIFAR10(root = filepath, train=False, download=True, transform = test_transformations)
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 40
    learning_rate = 0.001
    model = VGG16()
    #model.cuda()
    criterion = nn.CrossEntropyLoss()
    #RMSprop optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            # Forward Propogation
            #images = images.cuda()
            #labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward Propogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Accuracy test
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

        print('Epoch:{}, Loss:{:.4f}, Accuracy: {:.2f}%'
            .format(epoch + 1,  loss.item(), correct / total * 100.))

        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            #images, labels = Variable(images.cuda()),Variable(labels.cuda())
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Test Accuracy:{:.2f}%'.format(correct / total * 100.))