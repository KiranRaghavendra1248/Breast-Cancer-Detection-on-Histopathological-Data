# Imports
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import shutil
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from utils import getListOfFiles,check_accuracy, save_checkpoint, load_checkpoint, custom_transform, mean_std
from dataset import Breakhis


# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
shuffle_dataset = True
random_seed= 42
num_workers=0
learning_rate=0.001
print('Running on: ',device)
num_epochs=25
load_model=False
normalize = True



files=getListOfFiles('BreaKHis_v1')
imgs=[]
for f in files:
    if f.endswith('.png'):
        imgs.append(f)

# Create resnet model
model = models.resnet50(pretrained=False)
model.fc=nn.Sequential(nn.Linear(2048,1024),
                      nn.LeakyReLU(),
                      nn.Linear(1024,512),
                      nn.LeakyReLU(),
                      nn.Linear(512,2))
print(model)

# Loss and Optimizer
criterion=nn.CrossEntropyLoss()
optim=torch.optim.Adam(model.parameters(),lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('weights.pth.tar'),model,optim)

# Testing image processing result using a dummy transform
dummy_transform =transforms.Compose([
    transforms.Resize((512,512)),
    custom_transform()
])

# Test CLAHE
img=Image.open('BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-2523/100X/SOB_M_DC-14-2523-100-023.png')
img.save('before_transform.png')
result=dummy_transform(img)
plt.imsave('after_transform.png',result)

# Normalize Mean and STD over ALL images(test and train) using dummyset and dummyloader
if normalize:
    dummyset= Breakhis(imgs,transforms.Compose([
        transforms.Resize((521,521)),
        transforms.ToTensor()
    ]))
    print('Calculating mean and std for image normalization..')
    dummy_loader = DataLoader(dummyset, batch_size=batch_size,num_workers=num_workers, shuffle=True)
    mean,std=mean_std(dummy_loader,device)
    print('Mean: ',mean.tolist())
    print('Standard Deviation: ', std.tolist())

    # Include normalization and CLAHE in final transform
    transform =transforms.Compose([
        transforms.Resize((512,512)),
        custom_transform(),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(),std.tolist())
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        custom_transform(),
        transforms.ToTensor(),
    ])

dataset_normalized= Breakhis(imgs,transform)

# Random split into train test and validation
dataset_size=len(dataset_normalized)
print('Total images : ',dataset_size)
train_set,valid_set,test_set=random_split(dataset_normalized,[5539,1580,790])
print('Train, Validation, Test : ',len(train_set),len(valid_set),len(test_set))

# Create train and validation loader
train_loader = DataLoader(train_set, batch_size=batch_size,num_workers=num_workers, shuffle=True)
validation_loader = DataLoader(valid_set, batch_size=batch_size,num_workers=num_workers,shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size,num_workers=num_workers,shuffle=True)

# Put model on cuda
model.to(device)
# Put the model on train mode
model.train()
print()

# Test if o/p from model is of correct shape
i,y=next(iter(train_loader))
i=i.to(device)
y=y.to(device)
y_pred=model(i)
print(y_pred.shape)

# Training loop for the model
min_loss = None
for epoch in range(num_epochs):
    losses = []
    accuracies = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (data, labels) in loop:
        # Put data on cuda
        data = data.to(device)
        labels = labels.to(device).long()

        # Forward pass
        output = model(data)

        # Find out loss
        loss = criterion(output, labels)
        accuracy = check_accuracy(output, labels)
        losses.append(loss.detach().item())
        accuracies.append(accuracy.detach().item())

        optim.zero_grad()

        # Back prop
        loss.backward()

        # Step
        optim.step()

        # Update TQDM progress bar
        loop.set_description(f"Epoch [{epoch}/{num_epochs}] ")
        loop.set_postfix(loss=loss.detach().item(), accuracy=accuracy.detach().item())

    moving_loss = sum(losses) / len(losses)
    moving_accuracy = sum(accuracies) / len(accuracies)
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optim.state_dict()}
    # Save check point
    if min_loss == None:
        min_loss = moving_loss
        save_checkpoint(checkpoint)
    elif moving_loss < min_loss:
        min_loss = moving_loss
        save_checkpoint(checkpoint)
    print('Epoch {0} : Loss = {1} , Accuracy={2}'.format(epoch, moving_loss, moving_accuracy))


# Validation accuracy
print('Calculating validation accuracy..')
correct=0
samples=0
for data,labels in validation_loader:
    data=data.to(device)
    labels=labels.to(device)
    # Forward pass
    y_pred=model(data)
    # Accuracy over entire dataset
    _,predpos=y_pred.max(1)
    samples+=len(labels)
    correct+=(predpos==labels).sum().detach().item()
print('Validation accuracy : ',(correct/samples)*100)

# Test accuracy
correct=0
samples=0
print('Calculating test accuracy..')
for data,labels in test_loader:
    data=data.to(device)
    labels=labels.to(device)
    # Forward pass
    y_pred=model(data)
    # Accuracy over entire dataset
    _,predpos=y_pred.max(1)
    samples+=len(labels)
    correct+=(predpos==labels).sum().detach().item()
print('Test accuracy : ',(correct/samples)*100)