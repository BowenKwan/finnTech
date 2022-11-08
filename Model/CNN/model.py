# This scrit generates the onnx representation of the Quantized network
# This file can also be use to create train and test the FP model again the quantized version
# First setps to generate the hardware
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
# Bevitas
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling as BiasQuant
from brevitas.export import PyXIRManager
from brevitas.export import FINNManager
from data_preprocessing import *


parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, required=True)
parser.add_argument('--qdata', type=int, default=3)     #Default values set to 3
parser.add_argument('--qweight', type=int, default=3)
parser.add_argument('--qrelu', type=int, default=3)
args = parser.parse_args()

training = int(args.t)

if (training == 0):
    print("You have chosen option 0: The Network won't be trained ")
    training = 0
elif training == 1:
    print("You have chosen option 1: The Network will be trained")
else:
    # print in RED for the error message
    print('\n\x1b[1;31;40m' + 'Script Failed! -t is either 1 or 0' + '\x1b[0m')
    sys.exit()

#default values
# q_data_widt = 3
# q_bit_width = 3
# act_bit_width = 3

int(args.t)
q_data_widt = int(args.qdata)
q_bit_width = int(args.qweight)
act_bit_width = int(args.qrelu)

print("quantisation values for for the input data, the weight, and the relu are:", q_data_widt, q_bit_width, act_bit_width)

# Data loading
asset = 'AAPL'
assets = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']
level = 10
result_high = 0.9
result_low = 1-result_high

'''
[X, Y] = loadData('../../Data/{0}_2012-06-21_34200000_57600000', asset, level, result_high)
#[X,Y]=loadMultipleData('../../Data/{0}_2012-06-21_34200000_57600000', assets, level, result_high)


print(X)
print(Y)

# Normalise the INPUT value to improve the learning
X = preprocess(1)(X)

X_train_T, X_test_T, Y_train_T, Y_test_T = split_train_test_split(X, Y, 0.2)

print("X_train_T")
print(X_train_T)
print(X_train_T.shape)
'''



X_train_T, X_test_T, Y_train_T, Y_test_T =loadData2D('../../Data/{0}_2012-06-21_34200000_57600000', asset, level, result_high)


'''
X_train_T, X_test_T, Y_train_T, Y_test_T =loadMultipleData2D('../../Data/{0}_2012-06-21_34200000_57600000', assets, level, result_high)
'''

X_train_T=trimming_to_7(X_train_T)
X_test_T=trimming_to_7(X_test_T)

X_train_T=torch.unsqueeze(X_train_T,1)
X_test_T=torch.unsqueeze(X_test_T,1)

print("X_train_T")
#print(X_train_T[0,:])
#print(X_train_T[0,:,:])
print(X_train_T)
print(X_train_T.shape)

print("Y_train_T")
print(Y_train_T)
print(Y_train_T.shape)


# eport ONNX does not work with GPU data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device='cpu'
# transfer data to the GPU
X_train = X_train_T.to(device)
Y_train = Y_train_T.to(device)
X_test = X_test_T.to(device)
Y_test = Y_test_T.to(device)


print("X_train")
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

traindata=TensorDataset(X_train,Y_train)
testdata=TensorDataset(X_test,Y_test)
#print("traindata")
#print(traindata.shape)

trainLoader=DataLoader(dataset=traindata, shuffle=False)
testLoader=DataLoader(dataset=testdata,shuffle=False)


# saving npy files for verification process after transformation


quant_inp = qnn.QuantIdentity(bit_width=q_data_widt, return_quant_tensor=True)
quant_inp1 = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)


for i,(image, label) in list(enumerate(trainLoader))[:1]:
    print(image)
    input_np = quant_inp(image)
    print(input_np)
    input_np = quant_inp1(image)
    print(input_np)
    expected_output_np = label

print(input_np[0])
#input_np=input_np.to("cpu")
expected_output_np=expected_output_np.to("cpu")
if os.path.exists("./dataflow_build_dir"):
    shutil.rmtree("./dataflow_build_dir")  # delete pre-existing directory
    os.mkdir("./dataflow_build_dir")
    print("Previous run results deleted!")

else:
    os.mkdir("./dataflow_build_dir")

np.save('./dataflow_build_dir/input.npy', input_np[0])
np.save('./dataflow_build_dir/expected_output.npy', expected_output_np)
print("input and out npy have be SUCCESSFULLY genearted")
print(input_np)
print(input_np.shape)
print(expected_output_np)
print(expected_output_np.shape)

keep_prob = 1
# base network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
	    #size (?,level,7,1)
        self.layer1= nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)
        self.layer2= nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.fc1 = nn.Linear(512, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer1(x)
        #print("layer 1")
        #print(x.shape)
        x = self.layer2(x)
        #print("layer 2")
        #print(x.shape)
        x = self.layer3(x)
        #print("layer 3")
        #print(x.shape)
        #batched flattening
        #x = x.reshape(x.shape[0], -1)
        #unbatched flattening
        #x = x.reshape(1, -1)
        #print("level4")
        #print(x.shape)
        #x = self.fc1(x)
        #print("level5")
        #print(x.shape)
        #x = self.fc2(x)
        #print("level6")
        #print(x.shape)
        #print(x)
        x = torch.squeeze(x,0)
        #print("level7")
        #print(x.shape)
        #print(x)


        return (torch.sigmoid(x)*(result_high-result_low)+result_low)


#net = Net(X_train_T.shape[2])
net = Net()



# quantized network
# see examples https://github.com/Xilinx/brevitas


class QuantWeightNet(nn.Module):
    def __init__(self, n_features):
        super(QuantWeightNet, self).__init__()
        self.layer1= nn.Sequential(
            qnn.QuantConv2d(1, 32, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            #MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level,7,1)
        self.layer2= nn.Sequential(
            qnn.QuantConv2d(32, 64, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        #size (?,level/2,4,1)
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(64, 128, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=1 - keep_prob))
        self.fc1 = qnn.QuantLinear(512, 20,bias=True, weight_bit_width=q_bit_width)
        self.fc2 = qnn.QuantLinear(20, 1,bias=True, weight_bit_width=q_bit_width)

    def forward(self, x):
        #x = self.quant_inp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #batched flattening
        #x = x.reshape(x.shape[0], -1)
        #unbatched flattening
        #x = x.reshape(1, -1)
        #x = x.view(x.size[0], -1)
        #x=torch.flatten(x)
        #x = self.relu4(self.fc5(x))
        #x = self.fc1(x)
        #print("level5")
        #print(x.shape)
        #x = self.fc2(x)
        #print("level6")
        #print(x.shape)
        #print(x)
        #x = torch.squeeze(x,0)
        #return (torch.sigmoid(x)*(result_high-result_low)+result_low)
        return x
        # return ((self.fc3(x))*(result_high-result_low)+result_low) #Why is this stop ONNX export to work tested with 1 Layer


net_q = QuantWeightNet(X_train_T.shape[2])
# print(X_train_T.shape[1])
#print("NN structure is: ",net_q)
# print(net_q.parameters)


# Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities
criterion = nn.BCELoss(weight=None, size_average=None,
                       reduce=None, reduction='mean')

#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0003)
optimizer_q = optim.Adam(net_q.parameters(), lr=0.0003)

criterion = criterion.to(device)

net = net.to(device)
net_q = net_q.to(device)



'''
print("train loader")
print(trainLoader)
print(len(trainLoader.dataset))
for i,(image, label) in list(enumerate(trainLoader))[:1]:
    print(image)
    print(label)
print("test loader")
print(testLoader)
'''

if training == 1:
    print("\n", " TRAINING STARTING......".center(40, '#'), "\n")
    for epoch in range(2):
        running_loss=0.0
        correct = 0
        total = 0
        running_loss_q=0.0
        correct_q = 0
        total_q = 0
        #for i,(image, label) in (enumerate(trainLoader)):
        for i,(image, label) in list(enumerate(trainLoader))[:101]:
            label = torch.squeeze(label)
            #print("xXxxxxxxxxxxxx")
            #print(i)
            #print(image)
            #print(label)
            #print(image.shape)
            #print(label.shape)
            optimizer.zero_grad()  # reset Gradient descent
            optimizer_q.zero_grad()
            Y_pred = net(image)
            image_quant=quant_inp(image)
            Y_pred_q = net_q(image_quant)
            Y_pred = torch.squeeze(Y_pred)
            Y_pred_q = torch.squeeze(Y_pred_q)
            criterion = nn.BCELoss()
            train_loss = criterion(Y_pred, label)
            train_loss_q = criterion(Y_pred_q, label)
            train_loss.backward()  # propagate the error currently making
            optimizer.step()  # optimise
            train_loss_q.backward()  # propagate the error currently making
            optimizer_q.step()  # optimise

            running_loss += train_loss.item()
            running_loss_q += train_loss_q.item()
            
            total += 1
            predicted = int(Y_pred.item()>0.5)
            predicted_q = int(Y_pred_q.item()>0.5)
            #print("predicted")
            #print(Y_pred)
            #print(predicted)
            #print(label.item())
            tmp=int((label.item()>0.5) == (predicted))
            tmp_q=int((label.item()>0.5) == (predicted_q))
            #print('tmp')
            #print(tmp)
            correct += tmp
            correct_q += tmp_q
            #print("total")
            #print(total)
            #print(predicted)
            #print(tmp)
            #print(correct)

            if (i % 100000 == 0):
                running_loss_test=0.0
                correct_test = 0                
                total_test = 0
                running_loss_test_q=0.0
                correct_test_q = 0                
                total_test_q = 0
                #for j,(image_test, label_test) in enumerate(testLoader):
                for j,(image_test, label_test) in list(enumerate(testLoader))[:10]:
                    label_test = torch.squeeze(label_test)
                    Y_pred_test = net(image_test)
                    Y_pred_test = torch.squeeze(Y_pred_test)
                    image_test_quant=quant_inp(image_test)
            
                    Y_pred_test_q = net_q(image_test)
                    Y_pred_test_q = torch.squeeze(Y_pred_test_q)
                    criterion = nn.BCELoss()
                    test_loss = criterion(Y_pred_test, label_test)
                    test_loss_q = criterion(Y_pred_test_q, label_test)
                
                    running_loss_test += test_loss.item()   
                    running_loss_test_q += test_loss_q.item()                
                    total_test += 1
                    
                    predicted = int(Y_pred_test.item()>0.5)
                    predicted_q = int(Y_pred_test_q.item()>0.5)
                    
                    #print(j)
                    #print("predicted")
                    #print(Y_pred_test.item())
                    #print(predicted)
                    #print(label_test.item())
                    tmp=int((label_test.item()>0.5) == (predicted))
                    tmp_q=int((label_test.item()>0.5) == (predicted_q))
                    #print('tmp')
                    #print(tmp)
                    #print("total")
                    #print(total_test)
                    #print(predicted)
                    #print(tmp)
                    #print(correct_test)
                    correct_test += tmp
                    correct_test_q += tmp_q
                print("correct")
                print(correct)
                print(total)
                print("correct test")
                print(correct_test)
                print(total_test)
                print(f'''epoch {epoch} 
                Train set - loss: {running_loss/total}, accuracy: {float(correct)/total}                
                Test  set - loss: {running_loss_test/total_test}, accuracy: {float(correct_test)/total_test}
                ''')
                print("correct q")
                print(correct_q)
                print(total)
                print("correct test q")
                print(correct_test_q)
                print(total_test)
                print(f'''epoch {epoch} 
                Q Train set - loss: {running_loss_q/total}, accuracy: {float(correct_q)/total}                
                Q Test  set - loss: {running_loss_test_q/total_test}, accuracy: {float(correct_test_q)/total_test}
                ''')
else:
    print("\nNo training performed\n")


MODEL_PATH = 'model.pt'    # Full FP implementation
torch.save(net, MODEL_PATH)

print("Done saving net")

MODEL_PATH = 'model_q.pt'  # Quantized version of the full FP implemtation
torch.save(net_q.state_dict(), MODEL_PATH)

print("Done saving Q net")

print("\n", " GENERATING THE ONNX FILE ".center(40, '#'), "\n")

# Move to CPU before export
net_q.cpu()

# inp=torch.empty((1,1,1,7))
# inp = torch.randn(1, 1, 1, 7)
# inp = torch.randn(level, 7)
inp=input_np

FINNManager.export(net_q, input_t=inp, export_path='finn_QWNet2d_noFC.onnx')

#print('\x1b[6;30;42m' + 'Success!' + '\x1b[0m')
print("\x1b[6;30;42m" +
      "finn_QWNet2d.onnx has be SUCCESSFULLY genearted " + '\x1b[0m')

#import subprocess
#subprocess.call(["ls", "-l"])
os.system('ls -lt finn_QWNet2d.onnx')
