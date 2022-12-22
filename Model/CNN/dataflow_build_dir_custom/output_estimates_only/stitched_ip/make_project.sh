#!/bin/bash 
cd /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/vivado_stitch_proj_embwhftp
vivado -mode batch -source make_project.tcl
cd /scratch/bkwan/debug/finn/finnTech/Model/CNN/code_debug/dataflow_build_dir_custom
