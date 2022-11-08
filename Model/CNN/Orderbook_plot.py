#import pandas as pd
import numpy as np
import itertools
#from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import scipy.stats
#from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
#from hyperopt.pyll.base import scope
#import seaborn as sns
from matplotlib import cm, pyplot as plt
from scipy import stats as st
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
import torch.nn.functional as F
import torch
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling as BiasQuant
from brevitas.export import PyXIRManager

arr = np.loadtxt("../CNN result/resutls 2d 20 epoch train 1000 cuda 1 32 64 128 512 20.csv", delimiter=",")

arr = np.loadtxt("../CNN result/resutls 2d 20 epoch train 1000 cpu 10 enlarge 1 4 8 16 5184 200.csv", delimiter=",")
#arr = np.loadtxt("../CNN result/resutls 2d 20 epoch train 1000 cpu 1 instr level 10 time2d 1 4 8 16 1936 200.csv", delimiter=",")
#arr = np.loadtxt("../CNN result/resutls 2d 20 epoch train 500 cpu 5 instr level 10 time2d 1 8 16 32 23328 200 50.csv", delimiter=",")


for j in range(0,4):
    plot_l=plt.plot(range(0,20),arr[2*j,:])

plt.xlabel('training epoch') 
plt.ylabel('BCE loss') 

plt.xticks((0,5,10,15,20)) 
# displaying the title
plt.title("Training loss for enlarged 2d CNN Orderbook")
  

#plt.legend(['Float', 'Int32', 'Int16', 'Int12', 'Int8','Int6', 'Int4', 'Int3'],loc=3, fontsize='small', fancybox=True)
plt.legend(['Train', 'Q train', 'Test', 'Q test'],loc=3, fontsize='small', fancybox=True)

plt.savefig('2d 70x70 loss.png')

plt.show()
plt.close()


for j in range(0,4):
    plot_l=plt.plot(range(0,20),arr[1+2*j,:])

plt.xlabel('training epoch') 
plt.ylabel('Accuracy (%)') 
 
plt.xticks((0,5,10,15,20)) 
# displaying the title
plt.title("Accuracy for enlarge 2d CNN Orderbook")
  

plt.legend(['Train', 'Q train', 'Test', 'Q test'],loc=3, fontsize='small', fancybox=True)


plt.savefig('2d 70x70 acc.png')

plt.show()
plt.close()




