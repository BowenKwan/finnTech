##
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# CONFIDENTIAL - for internal use or use under binding NDA only
##

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline.absorb import AbsorbTransposeIntoMultiThreshold, AbsorbConsecutiveTransposes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast

def custom_step_convert_to_hls(model, cfg):
    model = model.transform(MakeInputChannelsLast())
    model = model.transform(AbsorbTransposeIntoMultiThreshold())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferVectorVectorActivation())
    model = model.transform(
        to_hls.InferQuantizedMatrixVectorActivation(mem_mode="decoupled")
    )
    print("\n","before infer bool".center(40, '#'), "\n")
    model = model.transform(to_hls.InferPool_Batch())
    print("\n","after infer bool".center(40, '#'), "\n")
    print("\n","before inferstreaming bool".center(40, '#'), "\n")
    model = model.transform(to_hls.InferStreamingMaxPool())
    print("\n","after inferstreaming bool".center(40, '#'), "\n")
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    return model
