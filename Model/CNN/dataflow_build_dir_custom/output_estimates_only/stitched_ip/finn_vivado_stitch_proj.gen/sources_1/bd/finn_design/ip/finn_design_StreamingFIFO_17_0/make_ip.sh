#!/bin/bash 
cd /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_17_jk4yjqhn/project_StreamingFIFO_17/sol1/impl/verilog
vivado -mode batch -source package_ip.tcl
cd /scratch/bkwan/debug/finn/finnTech/Model/CNN/code_debug/dataflow_build_dir_custom
