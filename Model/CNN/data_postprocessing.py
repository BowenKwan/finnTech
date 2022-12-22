#import onnx
#import torch
import os
from IPython.display import display
#from data_preProcessing import *
import pandas as pd
import argparse

# Analysing the output results of FINN compiler

# Directory structure


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()

name = args.name

estimates_output_dir = "dataflow_build_dir/output_estimates_only"

cmd = 'ls -l ./output_estimates_only'
os.system(cmd)
#print(f" ls -l ./{estimates_output_dir}/report")
print("\n", " ls -l ./output_estimates_only/report".center(80, '#'), "\n")
cmd = 'ls -l ./output_estimates_only/report'
os.system(cmd)
# Reading estimate_network_performance
print("\n", " Reading estimate_network_performance.json ".center(80, '#'), "\n")

# expect a list but the file does not contain the [value] so use the option dictionary
estimate_network_performance = pd.read_json(
    estimates_output_dir + '/report/estimate_network_performance.json', typ='dictionary')
display(estimate_network_performance)

# Reading estimate_layer_cycles
print("\n", " Reading estimate_layer_cycles.json ".center(80, '#'), "\n")
print("\nReading estimate_layer_cycles.json ")
estimate_layer_cycles = pd.read_json(
    estimates_output_dir + '/report/estimate_layer_cycles.json', typ='dictionary')
display(estimate_layer_cycles)

# Reading estimate_layer_resources
print("\n", " Reading estimate_layer_resources.json ".center(80, '#'), "\n")
estimate_layer_resources = pd.read_json(
    estimates_output_dir + '/report/estimate_layer_resources.json')
display(estimate_layer_resources)

print("\nestimated Latency\n")
print(estimate_network_performance['estimated_latency_ns'])
df = pd.DataFrame({'total': [estimate_network_performance['estimated_latency_ns']]}, index=['estimated_latency_ns'])
display(df) 



print("total FPGA resources")
df1 = estimate_layer_resources[['total']].copy()
display(df1)

# concatenating df1 and df2 along rows
# addin gthe latency information
result = pd.concat([df1, df], axis=0) #axis=0 column
print("total FPGA resources + estimated_latency_ns")
display(result)

#rename specific column names
result.rename(columns = {'total':'total_'+name}, inplace = True)
print("Change the name of the column to matach teh test case")
display(result)

#transpose the data for better visual
result=result.T

# Store the results in different file format for annalysis
if os.path.exists("./result/")==False:
    os.mkdir("./result/")

if os.path.exists("./result/results.csv"):
    result.to_csv('./result/results.csv',mode='a',index=True,header=False)
else:
    result.to_csv('./result/results.csv')

#result.to_csv('./results/results.csv', mode='a', index=False, header=False)
#result.to_excel('./results/resutls.xlsx')
#result.to_json(('./results/resutls.json'))
