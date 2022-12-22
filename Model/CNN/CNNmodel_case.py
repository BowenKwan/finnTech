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
from torchinfo import summary
# Bevitas
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling as BiasQuant
from brevitas.export import PyXIRManager
from brevitas.export import FINNManager
from data_preprocessing import *
from CNNModel import *

#parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, required=True)
parser.add_argument('--m', type=int, required=True)
parser.add_argument('--qdata', type=int, default=3)     #Default values set to 3
parser.add_argument('--qweight', type=int, default=3)
parser.add_argument('--qrelu', type=int, default=3)
args = parser.parse_args()


#setting for training the network or not
training = int(args.t)

if (training == 0):
    print("You have chosen option 0: The Network won't be trained ")
elif training == 1:
    print("You have chosen option 1: The Network will be trained")
else:
    # print in RED for the error message
    print('\n\x1b[1;31;40m' + 'Script Failed! -t is either 1 or 0' + '\x1b[0m')
    sys.exit()

#setting to choose model
model_num = int(args.m)
if (model_num == 1 or model_num == 2 or model_num == 3 or model_num == 4 or model_num == 5):
    print("You have chosen case %d"%(model_num))
else:
    # print in RED for the error message
    print('\n\x1b[1;31;40m' + 'Script Failed! -m has to be int from 1 to 5' + '\x1b[0m')
    sys.exit()



# set quantization bitwidth used  
q_data_widt = int(args.qdata)
q_bit_width = int(args.qweight)
act_bit_width = int(args.qrelu)

print("quantisation values for for the input data, the weight, and the relu are:", q_data_widt, q_bit_width, act_bit_width)

# quantization function
quant_inp = qnn.QuantIdentity(bit_width=q_data_widt, return_quant_tensor=True)

# Data loading
# asset name has to match with the csv file
filename='../../Data/{0}_2012-06-21_34200000_57600000'
asset = 'AAPL'
assets = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']   
level = 10      #level of orderbook used 
result_high = 1 #value indicating high, must be >0.5
result_low = 1 - result_high

X_train_T, X_test_T, Y_train_T, Y_test_T, net,net_q,model_struct=modelSelect(model_num)(filename,asset,assets,level,result_high,q_data_widt,q_bit_width,act_bit_width)


# eport ONNX does not work with GPU data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device ="cpu"
print(device)

# transfer data to the GPU
X_train = X_train_T.to(device)
Y_train = Y_train_T.to(device)
X_test = X_test_T.to(device)
Y_test = Y_test_T.to(device)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# compile dataloader for training and testing test
traindata=TensorDataset(X_train,Y_train)
testdata=TensorDataset(X_test,Y_test)

trainLoader=DataLoader(dataset=traindata, shuffle=False)
testLoader=DataLoader(dataset=testdata, shuffle=False)



# training parameter set up

# Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities
criterion = nn.BCELoss(weight=None, size_average=None,
                       reduce=None, reduction='mean')

learning_rate=0.001
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0003)
optimizer_q = optim.Adam(net_q.parameters(), lr=0.0003)

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
optimizer_q = optim.Adam(net_q.parameters(), lr=learning_rate)

print("network device")
print(device)
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

# get network architecture display with target input size
#summary(net,[1,7,10,10])

summary(net,X_train[0].unsqueeze(0).shape)

epoch_num=30
if training == 1:
    print("\n", " TRAINING STARTING......".center(40, '#'), "\n")
    cnn_train_loss=[]
    cnn_train_acc=[]
    cnn_train_loss_q=[]
    cnn_train_acc_q=[]

    cnn_test_loss=[]
    cnn_test_acc=[]
    cnn_test_loss_q=[]
    cnn_test_acc_q=[]

    for epoch in range(30):
        running_loss=0.0
        correct = 0
        total = 0
        running_loss_q=0.0
        correct_q = 0
        total_q = 0
        #for i,(image, label) in (enumerate(trainLoader)):
        for i,(image, label) in list(enumerate(trainLoader))[:10001]:
            label = torch.squeeze(label)
            #print("xXxxxxxxxxxxxx")
            #print(i)
            #print(image)
            #print(label)
            #print(label.shape)
            optimizer.zero_grad()  # reset Gradient descent
            optimizer_q.zero_grad()
            Y_pred = net(image)
            Y_pred_q = net_q(image)
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

            #if (i % 1000 == 0):
            #    print("\n", " TRAINING EPOCH "+ str(epoch)+ " i "+ str(i), "\n")
        running_loss_test=0.0
        correct_test = 0                
        total_test = 0
        running_loss_test_q=0.0
        correct_test_q = 0                
        total_test_q = 0
        #for j,(image_test, label_test) in enumerate(testLoader):
        for j,(image_test, label_test) in list(enumerate(testLoader))[:2501]:
            label_test = torch.squeeze(label_test)
            Y_pred_test = net(image_test)                                                         
            Y_pred_test = torch.squeeze(Y_pred_test)
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
        cnn_train_loss.append(running_loss/total)
        cnn_train_acc.append(float(correct)/total)
        cnn_train_loss_q.append(running_loss_q/total)
        cnn_train_acc_q.append(float(correct_q)/total)
        cnn_test_loss.append(running_loss_test/total_test)
        cnn_test_acc.append(float(correct_test)/total_test)
        cnn_test_loss_q.append(running_loss_test_q/total_test)
        cnn_test_acc_q.append(float(correct_test_q)/total_test)
else:
    print("\nNo training performed\n")

# save trained CNN model
# only saving model if training is done
if training == 1:
    MODEL_PATH = 'model_case%d.pt'%(model_num)    # Full FP implementation
    torch.save(net, MODEL_PATH)

    print("Done saving net")

    MODEL_PATH = 'model_case%d_q%d.pt'%(model_num,q_data_widt)  # Quantized version of the full FP implemtation
    torch.save(net_q.state_dict(), MODEL_PATH)

    print("Done saving Q net")

print("Done model saving")


# Generating ONNX file

# Move to CPU before export
net_q=net_q.cpu()

# saving npy files for verification process after transformation
for i,(image, label) in list(enumerate(trainLoader))[:1]:
    #get example input
    image=image.to("cpu")
    #quantize example input
    input_np = quant_inp(image)
    #get example output 
    expected_output_np = label

expected_output_np=expected_output_np.to("cpu")

# create build directory 
'''
if os.path.exists("./dataflow_build_dir"):
    shutil.rmtree("./dataflow_build_dir")  # delete pre-existing directory
    os.mkdir("./dataflow_build_dir")
    print("Previous run results deleted!")

else:
    os.mkdir("./dataflow_build_dir")
'''
if os.path.exists("./dataflow_build_dir")==False:
    os.mkdir("./dataflow_build_dir")

#input_np= torch.randn(1, 64*3*3)
#input_np=quant_inp(input_np)
#expected_output_np = 0

# need to run with FINN
# otherwise need to drop the grad on input_np[0] and input_np.scale before it can be saved 
np.save('./dataflow_build_dir/input.npy', input_np[0] / input_np.scale)
np.save('./dataflow_build_dir/expected_output.npy', expected_output_np)

np.save('./dataflow_build_dir_custom/input.npy', input_np[0] / input_np.scale)
np.save('./dataflow_build_dir_custom/expected_output.npy', expected_output_np)
print("input and out npy have be SUCCESSFULLY genearted")

                

inp=input_np
#inp=image
#inp= torch.randn(1, 64*3*3)
#print(inp.shape)
print(inp.shape)
FINNManager.export(net_q, input_t=inp, export_path='finn_QWNet2d_CNN_case%d.onnx'%(model_num))

#print('\x1b[6;30;42m' + 'Success!' + '\x1b[0m')
print("\x1b[6;30;42m" +
      "finn_QWNet2d_case%d.onnx has be SUCCESSFULLY genearted "%(model_num) + '\x1b[0m')

#import subprocess
#subprocess.call(["ls", "-l"])
os.system('ls -lt finn_QWNet2d_CNN_case%d.onnx'%(model_num))



# save acccuracy and loss data
if training == 1:
    displayResult(cnn_train_loss,cnn_train_acc,cnn_train_loss_q,cnn_train_acc_q,cnn_test_loss,cnn_test_acc,cnn_test_loss_q,cnn_test_acc_q)
    saveResult(model_num,model_struct,result_high,1,0,q_data_widt,learning_rate,cnn_train_loss,cnn_train_acc,cnn_train_loss_q,cnn_train_acc_q,cnn_test_loss,cnn_test_acc,cnn_test_loss_q,cnn_test_acc_q)

#print(net_q)
#print(net_q.fc2.weight)
#print(net_q.fc3.weight)
#print(net_q.parameters())

