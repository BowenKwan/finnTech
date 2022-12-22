import os
import sys
import shutil
import pandas as pd
import numpy as np
import argparse
import itertools
# sklearn
from sklearn.model_selection import train_test_split
# PyTorch
from torch import nn, optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader,TensorDataset
from torchinfo import summary
# Bevitas
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling as BiasQuant
from brevitas.export import PyXIRManager
from brevitas.export import FINNManager
from data_preprocessing import *

def modelSelect(case):
	return{
        1:model1,
		2:model2,
		3:model3,
		4:model4,
        5:model5,
	
	}.get(case,model_default)

def model_default():
    print("Incorrect model number, please choose from 1 to 5")
    return

def model1(filename,asset,assets,level,result_high,q_data_widt,q_bit_width,act_bit_width):
    #data parameter
        
    X_train_T, X_test_T, Y_train_T, Y_test_T =loadData2D(filename, asset, level, result_high)
    X_train_T=trimming_to_7(X_train_T)
    X_test_T=trimming_to_7(X_test_T)
    print("X_train_T")
    print(X_train_T)
    print(X_train_T.shape)

    X_train_T=torch.unsqueeze(X_train_T,1)
    X_test_T=torch.unsqueeze(X_test_T,1)

    #CNN parameter
    maxpool=2
    cnn1=16
    cnn2=32
    cnn3=64
    fully1=cnn3*2*2
    fully2=15
    fully3=1
    keep_prob = 0.5 # drop out rate= 1-keep_prob
    return X_train_T, X_test_T, Y_train_T, Y_test_T,Net1(maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,1-result_high), QNet1(maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,1-result_high,q_data_widt,q_bit_width,act_bit_width),"1 %d %d %d %d %d %d pool%d dropout %.2f"%(cnn1,cnn2,cnn3,fully1,fully2,fully3,maxpool,1-keep_prob)

# base network
class Net1(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,result_low):
        self.result_high=result_high
        self.result_low=result_low
        super(Net1, self).__init__()
	    #size (7,10,10)
        self.layer1= nn.Sequential(
            nn.Conv2d(1, cnn1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)  size (32,7,7)
        self.layer2= nn.Sequential(
            nn.Conv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)  size (64,3,3)
        self.layer3 = nn.Sequential(
            nn.Conv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        # size (128,2,2)
        self.fc1 = nn.Linear(fully1, fully2)  #768,28 for level 10
        self.fc2 = nn.Linear(fully2, fully3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)


# quantized network
# see examples https://github.com/Xilinx/brevitas

class QNet1(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,result_low,q_data_widt,q_bit_width,act_bit_width):
        self.result_high=result_high
        self.result_low=result_low
        super(QNet1, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=q_data_widt, return_quant_tensor=True)
        self.layer1= nn.Sequential(
            qnn.QuantConv2d(1, cnn1, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            #nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)
        self.layer2= nn.Sequential(
            qnn.QuantConv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.fc1 = qnn.QuantLinear(fully1, fully2,bias=True, weight_bit_width=q_bit_width)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width)
        self.fc2 = qnn.QuantLinear(fully2, fully3,bias=True, weight_bit_width=q_bit_width)
       
    def forward(self, x):
        x = self.quant_inp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(1, -1)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)

        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)


def model2(filename,asset,assets,level,result_high,q_data_widt,q_bit_width,act_bit_width):
    [X, Y] = loadData(filename, asset, level, result_high)
    X = preprocess(1)(X)

    X_train_T, X_test_T, Y_train_T, Y_test_T = split_train_test_split(X, Y, 0.2)
    print("X_train_T")
    print(X_train_T)
    print(X_train_T.shape)

    X_train_T=image1Dto2D(X_train_T)

    X_test_T=image1Dto2D(X_test_T)

    
    X_train_T=torch.unsqueeze(X_train_T,1)
    X_test_T=torch.unsqueeze(X_test_T,1)

    Y_train_T=Y_train_T[(X_train_T.shape[2]-1):]
    
    Y_test_T=Y_test_T[(X_test_T.shape[2]-1):]

    maxpool=2
    cnn1=32
    cnn2=64
    cnn3=128
    fully1=cnn3*6*6
    fully2=128
    fully3=10
    keep_prob = 0.5 # drop out rate= 1-keep_prob
    
    return X_train_T, X_test_T, Y_train_T, Y_test_T,Net2(maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,1-result_high), QNet2(maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,1-result_high,q_data_widt,q_bit_width,act_bit_width),"1 %d %d %d %d %d %d pool%d dropout %.2f"%(cnn1,cnn2,cnn3,fully1,fully2,fully3,maxpool,1-keep_prob)

# base network
class Net2(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,result_low):
        self.result_high=result_high
        self.result_low=result_low
        super(Net2, self).__init__()
	    #size (7,10,10)
        self.layer1= nn.Sequential(
            nn.Conv2d(1, cnn1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)  size (32,7,7)
        self.layer2= nn.Sequential(
            nn.Conv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)  size (64,3,3)
        self.layer3 = nn.Sequential(
            nn.Conv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        # size (128,2,2)
        self.fc1 = nn.Linear(fully1, fully2)  #768,28 for level 10
        self.fc2 = nn.Linear(fully2, fully3)
        self.fc3 = nn.Linear(fully3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)


# quantized network
# see examples https://github.com/Xilinx/brevitas

class QNet2(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,result_low,q_data_widt,q_bit_width,act_bit_width):
        self.result_high=result_high
        self.result_low=result_low
        super(QNet2, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=q_data_widt, return_quant_tensor=True)
        self.layer1= nn.Sequential(
            qnn.QuantConv2d(1, cnn1, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)
        self.layer2= nn.Sequential(
            qnn.QuantConv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.fc1 = qnn.QuantLinear(fully1, fully2,bias=True, weight_bit_width=q_bit_width)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width)
        self.fc2 = qnn.QuantLinear(fully2, fully3,bias=True, weight_bit_width=q_bit_width)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width)
        self.fc3 = qnn.QuantLinear(fully3, 1,bias=True, weight_bit_width=q_bit_width)
       
    def forward(self, x):
        x = self.quant_inp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(1, -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)

        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)



def model3(filename,asset,assets,level,result_high,q_data_widt,q_bit_width,act_bit_width):
    [X, Y] = loadMultipleData(filename, assets, level, result_high)
    X = preprocess(1)(X)

    X_train_T, X_test_T, Y_train_T, Y_test_T = split_train_test_split(X, Y, 0.2)
    print("X_train_T")
    print(X_train_T)
    print(X_train_T.shape)

    X_train_T=image1Dto2D(X_train_T)

    X_test_T=image1Dto2D(X_test_T)

    
    X_train_T=torch.unsqueeze(X_train_T,1)
    X_test_T=torch.unsqueeze(X_test_T,1)

    Y_train_T=Y_train_T[(X_train_T.shape[2]-1):]
    
    Y_test_T=Y_test_T[(X_test_T.shape[2]-1):]

    maxpool=2
    cnn1=64
    cnn2=128
    cnn3=256
    cnn4=256
    cnn5=256
    #5 instr level 10 parameter
    #fully1=cnn3*27*27
    #5 instr level 10 parameter pool3
    fully1=cnn3*8*8
    fully2=1000
    fully3=100
    keep_prob = 0.5 # drop out rate= 1-keep_prob
    
    return X_train_T, X_test_T, Y_train_T, Y_test_T,Net3(maxpool,cnn1,cnn2,cnn3,cnn4,cnn5,fully1,fully2,fully3,keep_prob,result_high,1-result_high), QNet3(maxpool,cnn1,cnn2,cnn3,cnn4,cnn5,fully1,fully2,fully3,keep_prob,result_high,1-result_high,q_data_widt,q_bit_width,act_bit_width),"1 %d %d %d %d %d %d pool%d dropout %.2f"%(cnn1,cnn2,cnn3,fully1,fully2,fully3,maxpool,1-keep_prob)

# base network
class Net3(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,cnn4,cnn5,fully1,fully2,fully3,keep_prob,result_high,result_low):
        self.result_high=result_high
        self.result_low=result_low
        super(Net3, self).__init__()
	    #size (7,10,10)
        self.layer1= nn.Sequential(
            nn.Conv2d(1, cnn1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)  size (32,7,7)
        self.layer2= nn.Sequential(
            nn.Conv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)  size (64,3,3)
        self.layer3 = nn.Sequential(
            nn.Conv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.layer4 = nn.Sequential(
            nn.Conv2d(cnn3, cnn4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.layer5 = nn.Sequential(
            nn.Conv2d(cnn4, cnn5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        # size (128,2,2)
        self.fc1 = nn.Linear(fully1, fully2)  #768,28 for level 10
        self.fc2 = nn.Linear(fully2, fully3)
        self.fc3 = nn.Linear(fully3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)


# quantized network
# see examples https://github.com/Xilinx/brevitas

class QNet3(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,cnn4,cnn5,fully1,fully2,fully3,keep_prob,result_high,result_low,q_data_widt,q_bit_width,act_bit_width):
        self.result_high=result_high
        self.result_low=result_low
        super(QNet3, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=q_data_widt, return_quant_tensor=True)
        self.layer1= nn.Sequential(
            qnn.QuantConv2d(1, cnn1, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)
        self.layer2= nn.Sequential(
            qnn.QuantConv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.layer4 = nn.Sequential(
            qnn.QuantConv2d(cnn3, cnn4, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.layer5 = nn.Sequential(
            qnn.QuantConv2d(cnn4, cnn5, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.fc1 = qnn.QuantLinear(fully1, fully2,bias=True, weight_bit_width=q_bit_width)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width)
        self.fc2 = qnn.QuantLinear(fully2, fully3,bias=True, weight_bit_width=q_bit_width)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width)
        self.fc3 = qnn.QuantLinear(fully3, 1,bias=True, weight_bit_width=q_bit_width)
       
    def forward(self, x):
        x = self.quant_inp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(1, -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)

        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)





#case 4
def model4(filename,asset,assets,level,result_high,q_data_widt,q_bit_width,act_bit_width):
    #data parameter
    timestep=10

    X_train_T, X_test_T, Y_train_T, Y_test_T =loadData2D_featureChannel(filename,asset,level,result_high,timestep)

    #CNN parameter
    maxpool=2
    cnn1=16
    cnn2=32
    cnn3=64
    fully1=cnn3*3*3
    fully2=fully1
    fully3=100
    keep_prob = 0.5 # drop out rate= 1-keep_prob
    return X_train_T, X_test_T, Y_train_T, Y_test_T,Net4(maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,1-result_high), QNet4(maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,1-result_high,q_data_widt,q_bit_width,act_bit_width),"7 %d %d %d %d %d %d pool%d dropout %.2f"%(cnn1,cnn2,cnn3,fully1,fully2,fully3,maxpool,1-keep_prob)

# base network
class Net4(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,result_low):
        self.result_high=result_high
        self.result_low=result_low
        super(Net4, self).__init__()
	    #size (7,10,10)
        self.layer1= nn.Sequential(
            nn.Conv2d(7, cnn1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn1, cnn1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn1, cnn1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (16,10,10)
        self.layer2= nn.Sequential(
            nn.Conv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn2, cnn2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn2, cnn2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (32,6,6)
        self.layer3 = nn.Sequential(
            nn.Conv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Conv2d(cnn3, cnn3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Conv2d(cnn3, cnn3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        # size (64,3,3) -> size(1,576)
        self.fc1 = nn.Linear(fully1, fully2)
        # size(1,576)
        self.fc2 = nn.Linear(fully2, fully3)
        # size (64,3,3) -> size(1,576)
        self.fc3 = nn.Linear(fully3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)


# quantized network
# see examples https://github.com/Xilinx/brevitas

class QNet4(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,result_low,q_data_widt,q_bit_width,act_bit_width):
        self.result_high=result_high
        self.result_low=result_low
        super(QNet4, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=q_data_widt, return_quant_tensor=True)
        self.layer1= nn.Sequential(
            qnn.QuantConv2d(7, cnn1, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn1, cnn1, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn1, cnn1, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            #nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)
        self.layer2= nn.Sequential(
            qnn.QuantConv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn2, cnn2, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn2, cnn2, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            qnn.QuantConv2d(cnn3, cnn3, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn3, cnn3, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.fc1 = qnn.QuantLinear(fully1, fully2,bias=True, weight_bit_width=q_bit_width)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width)
        self.fc2 = qnn.QuantLinear(fully2, fully3,bias=True, weight_bit_width=q_bit_width)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width)
        self.fc3 = qnn.QuantLinear(fully3, 1,bias=True, weight_bit_width=q_bit_width)
        #self.relu3 = qnn.QuantReLU(bit_width=act_bit_width)
        
    def forward(self, x):
        x = self.quant_inp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(1, -1)
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))       
        #x = self.relu3(self.fc3(x))
        x = self.fc3(x)

        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)

def model5(filename,asset,assets,level,result_high,q_data_widt,q_bit_width,act_bit_width):
        #data parameter
    timestep=50

        
    X_train_T, X_test_T, Y_train_T, Y_test_T =loadMultiData2D_featureChannel(filename, assets, level, result_high,timestep)

    print("X_train_T feature channel")
    print(X_train_T)
    print(X_train_T.shape)

    #CNN parameter
    maxpool=2
    cnn1=32
    cnn2=64
    cnn3=128
    fully1=cnn3*7*7
    fully2=fully1
    fully3=1000
    keep_prob = 0.5 # drop out rate= 1-keep_prob
    return X_train_T, X_test_T, Y_train_T, Y_test_T,Net5(maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,1-result_high), QNet5(maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,1-result_high,q_data_widt,q_bit_width,act_bit_width),"1 %d %d %d %d %d %d pool%d dropout %.2f"%(cnn1,cnn2,cnn3,fully1,fully2,fully3,maxpool,1-keep_prob)
# base network
class Net5(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,result_low):
        self.result_high=result_high
        self.result_low=result_low
        super(Net5, self).__init__()
	    #size (7,10,10)
        self.layer1= nn.Sequential(
            nn.Conv2d(7, cnn1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn1, cnn1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn1, cnn1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (16,10,10)
        self.layer2= nn.Sequential(
            nn.Conv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn2, cnn2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn2, cnn2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (32,6,6)
        self.layer3 = nn.Sequential(
            nn.Conv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Conv2d(cnn3, cnn3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Conv2d(cnn3, cnn3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        # size (64,3,3) -> size(1,576)
        self.fc1 = nn.Linear(fully1, fully2)
        # size(1,576)
        self.fc2 = nn.Linear(fully2, fully3)
        # size (64,3,3) -> size(1,576)
        self.fc3 = nn.Linear(fully3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)


# quantized network
# see examples https://github.com/Xilinx/brevitas

class QNet5(nn.Module):
    def __init__(self,maxpool,cnn1,cnn2,cnn3,fully1,fully2,fully3,keep_prob,result_high,result_low,q_data_widt,q_bit_width,act_bit_width):
        self.result_high=result_high
        self.result_low=result_low
        super(QNet5, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=q_data_widt, return_quant_tensor=True)
        self.layer1= nn.Sequential(
            qnn.QuantConv2d(7, cnn1, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn1, cnn1, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn1, cnn1, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)
        self.layer2= nn.Sequential(
            qnn.QuantConv2d(cnn1, cnn2, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn2, cnn2, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn2, cnn2, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(cnn2, cnn3, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            qnn.QuantConv2d(cnn3, cnn3, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),            
            qnn.QuantConv2d(cnn3, cnn3, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.fc1 = qnn.QuantLinear(fully1, fully2,bias=True, weight_bit_width=q_bit_width)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width)
        self.fc2 = qnn.QuantLinear(fully2, fully3,bias=True, weight_bit_width=q_bit_width)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width)
        self.fc3 = qnn.QuantLinear(fully3, 1,bias=True, weight_bit_width=q_bit_width)
        #self.relu3 = qnn.QuantReLU(bit_width=act_bit_width)
        
    def forward(self, x):
        x = self.quant_inp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(1, -1)
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))       
        #x = self.relu3(self.fc3(x))
        x = self.fc3(x)

        x = torch.squeeze(x,0)
        return (torch.sigmoid(x)*(self.result_high-self.result_low)+self.result_low)
