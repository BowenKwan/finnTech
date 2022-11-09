# Transform the ONNX file before sending it to the compiler
# 1. Import model into FINN with ModelWrapper
# 2. Genearting the /dataflow_build_dir/model.onnx"

import shutil
import sys
from time import time
import argparse
import finn.builder.build_dataflow_config as build_cfg
import finn.builder.build_dataflow as build
from finn.util.visualization import showInNetron
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.core.datatype import DataType
import onnx
import torch
import os
from pathlib import Path
from data_preprocessing import *
import json

from qonnx.core.modelwrapper import ModelWrapper

#file set up
parser = argparse.ArgumentParser()
parser.add_argument('--f', type=str, required=True)
args = parser.parse_args()

pathName, fileName = os.path.split(args.f)  # remove the path name
ready_model_filename = fileName

print(" the name of the model_for_sim is", ready_model_filename)

model_for_sim = ModelWrapper(ready_model_filename)


dir(model_for_sim)

#get onnx info
finnonnx_in_tensor_name = model_for_sim.graph.input[0].name
finnonnx_out_tensor_name = model_for_sim.graph.output[0].name
print("Input tensor name: %s" % finnonnx_in_tensor_name)
print("Output tensor name: %s" % finnonnx_out_tensor_name)
finnonnx_model_in_shape = model_for_sim.get_tensor_shape(
    finnonnx_in_tensor_name)
finnonnx_model_out_shape = model_for_sim.get_tensor_shape(
    finnonnx_out_tensor_name)
print("Input tensor shape: %s" % str(finnonnx_model_in_shape))
print("Output tensor shape: %s" % str(finnonnx_model_out_shape))
finnonnx_model_in_dt = model_for_sim.get_tensor_datatype(
    finnonnx_in_tensor_name)
finnonnx_model_out_dt = model_for_sim.get_tensor_datatype(
    finnonnx_out_tensor_name)
print("Input tensor datatype: %s" % str(finnonnx_model_in_dt.name))
print("Output tensor datatype: %s" % str(finnonnx_model_out_dt.name))
print("List of node operator types in the graph: ")
print([x.op_type for x in model_for_sim.graph.node])

# 2. Network preparation: Tidy-up transformations
# Before running the verification, we need to prepare our FINN-ONNX model. In particular,
# all the intermediate tensors need to have statically defined shapes.
# To do this, we apply some graph transformations to the model like a kind of "tidy-up" to make it easier to process.

model_for_sim = model_for_sim.transform(InferShapes())
model_for_sim = model_for_sim.transform(FoldConstants())
model_for_sim = model_for_sim.transform(GiveUniqueNodeNames())
model_for_sim = model_for_sim.transform(GiveReadableTensorNames())
model_for_sim = model_for_sim.transform(InferDataTypes())
model_for_sim = model_for_sim.transform(RemoveStaticGraphInputs())

#verif_model_filename = "finn_QWNet111-verification.onnx"
# create a new name based on the input name
verif_model_filename = Path(ready_model_filename).stem + "-verification.onnx"

# NAME IS FIXED CAN BE CHANGED NEEDED BY THE FINN COMPILER

if os.path.exists("./onnx_models/"):
    shutil.rmtree("./onnx_models/")  # delete pre-existing directory
    os.mkdir("./onnx_models/")
    print("Previous run results deleted!")

else:
    os.mkdir("./onnx_models/")

model_for_sim.save("./onnx_models/" + verif_model_filename)
model_for_sim.save("./dataflow_build_dir/model.onnx")  # fixed by the compiler

# Copy the configuration file for the FINN compiler
shutil.copyfile("../configuration_files/dataflow_build_config.json",
                "./dataflow_build_dir/dataflow_build_config.json")

print("NEXT STEP: use the storeResult.ipynb to visualize and save the resutls")

print("\n", " python ../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir ".center(80, '#'), "\n")
