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

# saving npy files for verification process after transformation
[trainLoader,testLoader]=load_dataset('../../../Data/{0}_2012-06-21_34200000_57600000', asset, level, result_high)

quant_inp = qnn.QuantIdentity(bit_width=q_data_widt, return_quant_tensor=True)
quant_inp1 = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)

keep_prob = 1

# quantized network
# see examples https://github.com/Xilinx/brevitas


class QuantWeightNet(nn.Module):
    def __init__(self):
        super(QuantWeightNet, self).__init__()
        self.layer1= nn.Sequential(
            qnn.QuantConv2d(1, 32, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            #QuantMaxPool2d(kernel_size=2, return_quant_tensor=False)
            #nn.Dropout(p=1 - keep_prob)
            )
        #size (?,level,7,1)
        self.layer2= nn.Sequential(
            qnn.QuantConv2d(32, 64, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            qnn.QuantMaxPool2d(kernel_size=2, return_quant_tensor=False),
            #nn.Dropout(p=1 - keep_prob)
            )
        #size (?,level/2,4,1)
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(64, 128, kernel_size=3, stride=1, padding=1,weight_bit_width=q_bit_width),
            qnn.QuantReLU(),
            qnn.QuantMaxPool2d(kernel_size=2, return_quant_tensor=False),
            #nn.Dropout(p=1 - keep_prob)
            )
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


net_q = QuantWeightNet()

# Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities
criterion = nn.BCELoss(weight=None, size_average=None,
                       reduce=None, reduction='mean')

optimizer_q = optim.Adam(net_q.parameters(), lr=0.0003)

criterion = criterion.to(device)

net_q = net_q.to(device)

if training == 1:
    print("\n", " TRAINING STARTING......".center(40, '#'), "\n")
    for epoch in range(2):
        #for i,(image, label) in (enumerate(trainLoader)):
        for i,(image, label) in list(enumerate(trainLoader))[:101]:
            label = torch.squeeze(label)
             # reset Gradient descent
            optimizer_q.zero_grad()
            image_quant=quant_inp(image)
            Y_pred_q = net_q(image_quant)
            Y_pred_q = torch.squeeze(Y_pred_q)
            train_loss_q = criterion(Y_pred_q, label)
            train_loss_q.backward()  # propagate the error currently making
            optimizer_q.step()  # optimise
else:
    print("\nNo training performed\n")

MODEL_PATH = 'model_q.pt'  # Quantized version of the full FP implemtation
torch.save(net_q.state_dict(), MODEL_PATH)

print("Done saving Q net")

print("\n", " GENERATING THE ONNX FILE ".center(40, '#'), "\n")

# Move to CPU before export
net_q.cpu()

# inp=torch.empty((1,1,1,7))
# inp = torch.randn(1, 1, 1, 7)
# inp = torch.randn(level, 7)

for i,(image, label) in list(enumerate(trainLoader))[:1]:
    print(image)
    input_np = quant_inp1(image)
    print(input_np)
    expected_output_np = label

expected_output_np=expected_output_np.to("cpu")
if os.path.exists("./dataflow_build_dir"):
    shutil.rmtree("./dataflow_build_dir")  # delete pre-existing directory
    os.mkdir("./dataflow_build_dir")
    print("Previous run results deleted!")

else:
    os.mkdir("./dataflow_build_dir")
np.save('./dataflow_build_dir/input.npy', input_np[0] / input_np.scale)
np.save('./dataflow_build_dir/expected_output.npy', expected_output_np)
print("input and out npy have be SUCCESSFULLY genearted")

inp=input_np

FINNManager.export(net_q, input_t=inp, export_path='finn_QWNet2d_noFC.onnx')

#print('\x1b[6;30;42m' + 'Success!' + '\x1b[0m')
print("\x1b[6;30;42m" +
      "finn_QWNet2d.onnx has be SUCCESSFULLY genearted " + '\x1b[0m')

#import subprocess
#subprocess.call(["ls", "-l"])
os.system('ls -lt finn_QWNet2d.onnx')
