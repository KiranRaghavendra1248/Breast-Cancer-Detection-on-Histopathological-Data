
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
from preprocess import clahe

# Get list of all files in directory
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

# Check accuracy function
def check_accuracy(output,labels):
    _,predpos=output.max(1)
    num_samples=len(labels)
    num_correct=(predpos==labels).sum()
    return (num_correct/num_samples)*100

# Function to calc mean and std across dataset
def mean_std(loader,device):
    # V(X) = E(X**2)-E(X)**2
    channels_sum,channels_squared_sum,num_batches = 0, 0, 0
    for data,_ in loader:
        data.to(device)
        channels_sum+=torch.mean(data,[0,2,3])
        channels_squared_sum+=torch.mean(data**2,[0,2,3])
        num_batches+=1
    mean=channels_sum/num_batches
    std=(channels_squared_sum/num_batches-mean**2)**0.5
    return mean,std

# Save checkpoint
def save_checkpoint(state,filename='weights.pth.tar'):
    print('Saving weights-->')
    torch.save(state,filename)

# Load checkpoint
def load_checkpoint(checkpoint,model,optim):
    print('Loading weights-->')
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])


# Create custom transform using preprocessing function
class custom_transform():
    def __call__(self, img):
        return clahe(img)

    def __repr__(self):
        print('CLAHE()')