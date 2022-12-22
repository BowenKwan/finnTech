create_project finn_vivado_stitch_proj /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/vivado_stitch_proj_embwhftp -part xc7z020clg400-1
set_property ip_repo_paths [list $::env(FINN_ROOT)/finn-rtllib/memstream /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_0_5zhef_dm/project_StreamingFIFO_0/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_FMPadding_Batch_0__sj4l1ob/project_FMPadding_Batch_0/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_1_11_0zpih/project_StreamingFIFO_1/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_ConvolutionInputGenerator_0_9ctwy97f/project_ConvolutionInputGenerator_0/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingDataWidthConverter_Batch_0__u5uurwx/project_StreamingDataWidthConverter_Batch_0/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_3_6ailiv0a/project_StreamingFIFO_3/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_MatrixVectorActivation_0_wx4gew9p/project_MatrixVectorActivation_0/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_Thresholding_Batch_0_84gyb68r/project_Thresholding_Batch_0/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingDataWidthConverter_Batch_1__wvvdb40/project_StreamingDataWidthConverter_Batch_1/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_6_bjxxxy20/project_StreamingFIFO_6/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_FMPadding_Batch_1_g0eos47s/project_FMPadding_Batch_1/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingDataWidthConverter_Batch_2_hox5tppd/project_StreamingDataWidthConverter_Batch_2/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_8_tq6ibsv0/project_StreamingFIFO_8/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_ConvolutionInputGenerator_1_sadv8ch7/project_ConvolutionInputGenerator_1/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingDataWidthConverter_Batch_3_bzps61sn/project_StreamingDataWidthConverter_Batch_3/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_MatrixVectorActivation_1__w__vqmd/project_MatrixVectorActivation_1/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_Thresholding_Batch_1_xzl9ip90/project_Thresholding_Batch_1/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_12_mwrcb11o/project_StreamingFIFO_12/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_ConvolutionInputGenerator_2_afn4bzfp/project_ConvolutionInputGenerator_2/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_Pool_Batch_0_qrlgd4ji/project_Pool_Batch_0/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingDataWidthConverter_Batch_4_3z2a3gsk/project_StreamingDataWidthConverter_Batch_4/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_15_jd6yapy3/project_StreamingFIFO_15/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_FMPadding_Batch_2_3nethm_z/project_FMPadding_Batch_2/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingDataWidthConverter_Batch_5_9x2k9lsw/project_StreamingDataWidthConverter_Batch_5/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_17_jk4yjqhn/project_StreamingFIFO_17/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_ConvolutionInputGenerator_3_r_7hxvut/project_ConvolutionInputGenerator_3/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingDataWidthConverter_Batch_6_o9n40t9o/project_StreamingDataWidthConverter_Batch_6/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_MatrixVectorActivation_2_g310_uuv/project_MatrixVectorActivation_2/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_Thresholding_Batch_2__scv1oh_/project_Thresholding_Batch_2/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingDataWidthConverter_Batch_7_x5jurtsn/project_StreamingDataWidthConverter_Batch_7/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_22_lmm19yzr/project_StreamingFIFO_22/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_FMPadding_Batch_3_uujts_j8/project_FMPadding_Batch_3/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_23_4msqbo_a/project_StreamingFIFO_23/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_ConvolutionInputGenerator_4_j87_psdh/project_ConvolutionInputGenerator_4/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_Pool_Batch_1_p9j_sujn/project_Pool_Batch_1/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingDataWidthConverter_Batch_8_9by0yzz1/project_StreamingDataWidthConverter_Batch_8/sol1/impl/ip /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_StreamingFIFO_26_dpamh9qq/project_StreamingFIFO_26/sol1/impl/verilog /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_MatrixVectorActivation_3_w6za0nuu/project_MatrixVectorActivation_3/sol1/impl/ip] [current_project]
update_ip_catalog
create_bd_design "finn_design"
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingFIFO_0:1.0 StreamingFIFO_0
create_bd_cell -type ip -vlnv xilinx.com:hls:FMPadding_Batch_0:1.0 FMPadding_Batch_0
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingFIFO_1:1.0 StreamingFIFO_1
create_bd_cell -type ip -vlnv xilinx.com:hls:ConvolutionInputGenerator_0:1.0 ConvolutionInputGenerator_0
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingDataWidthConverter_Batch_0:1.0 StreamingDataWidthConverter_Batch_0
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingFIFO_3:1.0 StreamingFIFO_3
create_bd_cell -type hier MatrixVectorActivation_0
create_bd_pin -dir I -type clk /MatrixVectorActivation_0/ap_clk
create_bd_pin -dir I -type rst /MatrixVectorActivation_0/ap_rst_n
create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 /MatrixVectorActivation_0/out_V
create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 /MatrixVectorActivation_0/in0_V
create_bd_cell -type ip -vlnv xilinx.com:hls:MatrixVectorActivation_0:1.0 /MatrixVectorActivation_0/MatrixVectorActivation_0
create_bd_cell -type ip -vlnv xilinx.com:user:memstream:1.0 /MatrixVectorActivation_0/MatrixVectorActivation_0_wstrm
set_property -dict [list CONFIG.NSTREAMS {1} CONFIG.MEM_DEPTH {16} CONFIG.MEM_WIDTH {72} CONFIG.MEM_INIT {/scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_MatrixVectorActivation_0_wx4gew9p/} CONFIG.RAM_STYLE {auto} CONFIG.STRM0_DEPTH {16} CONFIG.STRM0_WIDTH {72} CONFIG.STRM0_OFFSET {0} ] [get_bd_cells /MatrixVectorActivation_0/MatrixVectorActivation_0_wstrm]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_0/MatrixVectorActivation_0_wstrm/m_axis_0] [get_bd_intf_pins MatrixVectorActivation_0/MatrixVectorActivation_0/weights_V]
connect_bd_net [get_bd_pins MatrixVectorActivation_0/ap_rst_n] [get_bd_pins MatrixVectorActivation_0/MatrixVectorActivation_0_wstrm/aresetn]
connect_bd_net [get_bd_pins MatrixVectorActivation_0/ap_clk] [get_bd_pins MatrixVectorActivation_0/MatrixVectorActivation_0_wstrm/aclk]
connect_bd_net [get_bd_pins MatrixVectorActivation_0/ap_rst_n] [get_bd_pins MatrixVectorActivation_0/MatrixVectorActivation_0/ap_rst_n]
connect_bd_net [get_bd_pins MatrixVectorActivation_0/ap_clk] [get_bd_pins MatrixVectorActivation_0/MatrixVectorActivation_0/ap_clk]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_0/in0_V] [get_bd_intf_pins MatrixVectorActivation_0/MatrixVectorActivation_0/in0_V]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_0/out_V] [get_bd_intf_pins MatrixVectorActivation_0/MatrixVectorActivation_0/out_V]
save_bd_design
create_bd_cell -type ip -vlnv xilinx.com:hls:Thresholding_Batch_0:1.0 Thresholding_Batch_0
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingDataWidthConverter_Batch_1:1.0 StreamingDataWidthConverter_Batch_1
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingFIFO_6:1.0 StreamingFIFO_6
create_bd_cell -type ip -vlnv xilinx.com:hls:FMPadding_Batch_1:1.0 FMPadding_Batch_1
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingDataWidthConverter_Batch_2:1.0 StreamingDataWidthConverter_Batch_2
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingFIFO_8:1.0 StreamingFIFO_8
create_bd_cell -type ip -vlnv xilinx.com:hls:ConvolutionInputGenerator_1:1.0 ConvolutionInputGenerator_1
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingDataWidthConverter_Batch_3:1.0 StreamingDataWidthConverter_Batch_3
create_bd_cell -type hier MatrixVectorActivation_1
create_bd_pin -dir I -type clk /MatrixVectorActivation_1/ap_clk
create_bd_pin -dir I -type rst /MatrixVectorActivation_1/ap_rst_n
create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 /MatrixVectorActivation_1/out_V
create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 /MatrixVectorActivation_1/in0_V
create_bd_cell -type ip -vlnv xilinx.com:hls:MatrixVectorActivation_1:1.0 /MatrixVectorActivation_1/MatrixVectorActivation_1
create_bd_cell -type ip -vlnv xilinx.com:user:memstream:1.0 /MatrixVectorActivation_1/MatrixVectorActivation_1_wstrm
set_property -dict [list CONFIG.NSTREAMS {1} CONFIG.MEM_DEPTH {16} CONFIG.MEM_WIDTH {2304} CONFIG.MEM_INIT {/scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_MatrixVectorActivation_1__w__vqmd/} CONFIG.RAM_STYLE {auto} CONFIG.STRM0_DEPTH {16} CONFIG.STRM0_WIDTH {2304} CONFIG.STRM0_OFFSET {0} ] [get_bd_cells /MatrixVectorActivation_1/MatrixVectorActivation_1_wstrm]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_1/MatrixVectorActivation_1_wstrm/m_axis_0] [get_bd_intf_pins MatrixVectorActivation_1/MatrixVectorActivation_1/weights_V]
connect_bd_net [get_bd_pins MatrixVectorActivation_1/ap_rst_n] [get_bd_pins MatrixVectorActivation_1/MatrixVectorActivation_1_wstrm/aresetn]
connect_bd_net [get_bd_pins MatrixVectorActivation_1/ap_clk] [get_bd_pins MatrixVectorActivation_1/MatrixVectorActivation_1_wstrm/aclk]
connect_bd_net [get_bd_pins MatrixVectorActivation_1/ap_rst_n] [get_bd_pins MatrixVectorActivation_1/MatrixVectorActivation_1/ap_rst_n]
connect_bd_net [get_bd_pins MatrixVectorActivation_1/ap_clk] [get_bd_pins MatrixVectorActivation_1/MatrixVectorActivation_1/ap_clk]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_1/in0_V] [get_bd_intf_pins MatrixVectorActivation_1/MatrixVectorActivation_1/in0_V]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_1/out_V] [get_bd_intf_pins MatrixVectorActivation_1/MatrixVectorActivation_1/out_V]
save_bd_design
create_bd_cell -type ip -vlnv xilinx.com:hls:Thresholding_Batch_1:1.0 Thresholding_Batch_1
create_bd_cell -type hier StreamingFIFO_12
create_bd_pin -dir I -type clk /StreamingFIFO_12/ap_clk
create_bd_pin -dir I -type rst /StreamingFIFO_12/ap_rst_n
create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 /StreamingFIFO_12/out_V
create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 /StreamingFIFO_12/in0_V
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 /StreamingFIFO_12/fifo
set_property -dict [list CONFIG.FIFO_DEPTH {512}] [get_bd_cells /StreamingFIFO_12/fifo]
set_property -dict [list CONFIG.FIFO_MEMORY_TYPE {auto}] [get_bd_cells /StreamingFIFO_12/fifo]
set_property -dict [list CONFIG.TDATA_NUM_BYTES {2}] [get_bd_cells /StreamingFIFO_12/fifo]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_12/fifo/M_AXIS] [get_bd_intf_pins StreamingFIFO_12/out_V]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_12/fifo/S_AXIS] [get_bd_intf_pins StreamingFIFO_12/in0_V]
connect_bd_net [get_bd_pins StreamingFIFO_12/ap_rst_n] [get_bd_pins StreamingFIFO_12/fifo/s_axis_aresetn]
connect_bd_net [get_bd_pins StreamingFIFO_12/ap_clk] [get_bd_pins StreamingFIFO_12/fifo/s_axis_aclk]
create_bd_cell -type ip -vlnv xilinx.com:hls:ConvolutionInputGenerator_2:1.0 ConvolutionInputGenerator_2
create_bd_cell -type ip -vlnv xilinx.com:hls:Pool_Batch_0:1.0 Pool_Batch_0
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingDataWidthConverter_Batch_4:1.0 StreamingDataWidthConverter_Batch_4
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingFIFO_15:1.0 StreamingFIFO_15
create_bd_cell -type ip -vlnv xilinx.com:hls:FMPadding_Batch_2:1.0 FMPadding_Batch_2
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingDataWidthConverter_Batch_5:1.0 StreamingDataWidthConverter_Batch_5
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingFIFO_17:1.0 StreamingFIFO_17
create_bd_cell -type ip -vlnv xilinx.com:hls:ConvolutionInputGenerator_3:1.0 ConvolutionInputGenerator_3
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingDataWidthConverter_Batch_6:1.0 StreamingDataWidthConverter_Batch_6
create_bd_cell -type hier MatrixVectorActivation_2
create_bd_pin -dir I -type clk /MatrixVectorActivation_2/ap_clk
create_bd_pin -dir I -type rst /MatrixVectorActivation_2/ap_rst_n
create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 /MatrixVectorActivation_2/out_V
create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 /MatrixVectorActivation_2/in0_V
create_bd_cell -type ip -vlnv xilinx.com:hls:MatrixVectorActivation_2:1.0 /MatrixVectorActivation_2/MatrixVectorActivation_2
create_bd_cell -type ip -vlnv xilinx.com:user:memstream:1.0 /MatrixVectorActivation_2/MatrixVectorActivation_2_wstrm
set_property -dict [list CONFIG.NSTREAMS {1} CONFIG.MEM_DEPTH {64} CONFIG.MEM_WIDTH {2304} CONFIG.MEM_INIT {/scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_MatrixVectorActivation_2_g310_uuv/} CONFIG.RAM_STYLE {auto} CONFIG.STRM0_DEPTH {64} CONFIG.STRM0_WIDTH {2304} CONFIG.STRM0_OFFSET {0} ] [get_bd_cells /MatrixVectorActivation_2/MatrixVectorActivation_2_wstrm]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_2/MatrixVectorActivation_2_wstrm/m_axis_0] [get_bd_intf_pins MatrixVectorActivation_2/MatrixVectorActivation_2/weights_V]
connect_bd_net [get_bd_pins MatrixVectorActivation_2/ap_rst_n] [get_bd_pins MatrixVectorActivation_2/MatrixVectorActivation_2_wstrm/aresetn]
connect_bd_net [get_bd_pins MatrixVectorActivation_2/ap_clk] [get_bd_pins MatrixVectorActivation_2/MatrixVectorActivation_2_wstrm/aclk]
connect_bd_net [get_bd_pins MatrixVectorActivation_2/ap_rst_n] [get_bd_pins MatrixVectorActivation_2/MatrixVectorActivation_2/ap_rst_n]
connect_bd_net [get_bd_pins MatrixVectorActivation_2/ap_clk] [get_bd_pins MatrixVectorActivation_2/MatrixVectorActivation_2/ap_clk]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_2/in0_V] [get_bd_intf_pins MatrixVectorActivation_2/MatrixVectorActivation_2/in0_V]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_2/out_V] [get_bd_intf_pins MatrixVectorActivation_2/MatrixVectorActivation_2/out_V]
save_bd_design
create_bd_cell -type ip -vlnv xilinx.com:hls:Thresholding_Batch_2:1.0 Thresholding_Batch_2
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingDataWidthConverter_Batch_7:1.0 StreamingDataWidthConverter_Batch_7
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingFIFO_22:1.0 StreamingFIFO_22
create_bd_cell -type ip -vlnv xilinx.com:hls:FMPadding_Batch_3:1.0 FMPadding_Batch_3
create_bd_cell -type hier StreamingFIFO_23
create_bd_pin -dir I -type clk /StreamingFIFO_23/ap_clk
create_bd_pin -dir I -type rst /StreamingFIFO_23/ap_rst_n
create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 /StreamingFIFO_23/out_V
create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 /StreamingFIFO_23/in0_V
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 /StreamingFIFO_23/fifo
set_property -dict [list CONFIG.FIFO_DEPTH {1024}] [get_bd_cells /StreamingFIFO_23/fifo]
set_property -dict [list CONFIG.FIFO_MEMORY_TYPE {auto}] [get_bd_cells /StreamingFIFO_23/fifo]
set_property -dict [list CONFIG.TDATA_NUM_BYTES {2}] [get_bd_cells /StreamingFIFO_23/fifo]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_23/fifo/M_AXIS] [get_bd_intf_pins StreamingFIFO_23/out_V]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_23/fifo/S_AXIS] [get_bd_intf_pins StreamingFIFO_23/in0_V]
connect_bd_net [get_bd_pins StreamingFIFO_23/ap_rst_n] [get_bd_pins StreamingFIFO_23/fifo/s_axis_aresetn]
connect_bd_net [get_bd_pins StreamingFIFO_23/ap_clk] [get_bd_pins StreamingFIFO_23/fifo/s_axis_aclk]
create_bd_cell -type ip -vlnv xilinx.com:hls:ConvolutionInputGenerator_4:1.0 ConvolutionInputGenerator_4
create_bd_cell -type ip -vlnv xilinx.com:hls:Pool_Batch_1:1.0 Pool_Batch_1
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingDataWidthConverter_Batch_8:1.0 StreamingDataWidthConverter_Batch_8
create_bd_cell -type ip -vlnv xilinx.com:hls:StreamingFIFO_26:1.0 StreamingFIFO_26
create_bd_cell -type hier MatrixVectorActivation_3
create_bd_pin -dir I -type clk /MatrixVectorActivation_3/ap_clk
create_bd_pin -dir I -type rst /MatrixVectorActivation_3/ap_rst_n
create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 /MatrixVectorActivation_3/out_V
create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 /MatrixVectorActivation_3/in0_V
create_bd_cell -type ip -vlnv xilinx.com:hls:MatrixVectorActivation_3:1.0 /MatrixVectorActivation_3/MatrixVectorActivation_3
create_bd_cell -type ip -vlnv xilinx.com:user:memstream:1.0 /MatrixVectorActivation_3/MatrixVectorActivation_3_wstrm
set_property -dict [list CONFIG.NSTREAMS {1} CONFIG.MEM_DEPTH {960} CONFIG.MEM_WIDTH {32} CONFIG.MEM_INIT {/scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_MatrixVectorActivation_3_w6za0nuu/} CONFIG.RAM_STYLE {auto} CONFIG.STRM0_DEPTH {960} CONFIG.STRM0_WIDTH {32} CONFIG.STRM0_OFFSET {0} ] [get_bd_cells /MatrixVectorActivation_3/MatrixVectorActivation_3_wstrm]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_3/MatrixVectorActivation_3_wstrm/m_axis_0] [get_bd_intf_pins MatrixVectorActivation_3/MatrixVectorActivation_3/weights_V]
connect_bd_net [get_bd_pins MatrixVectorActivation_3/ap_rst_n] [get_bd_pins MatrixVectorActivation_3/MatrixVectorActivation_3_wstrm/aresetn]
connect_bd_net [get_bd_pins MatrixVectorActivation_3/ap_clk] [get_bd_pins MatrixVectorActivation_3/MatrixVectorActivation_3_wstrm/aclk]
connect_bd_net [get_bd_pins MatrixVectorActivation_3/ap_rst_n] [get_bd_pins MatrixVectorActivation_3/MatrixVectorActivation_3/ap_rst_n]
connect_bd_net [get_bd_pins MatrixVectorActivation_3/ap_clk] [get_bd_pins MatrixVectorActivation_3/MatrixVectorActivation_3/ap_clk]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_3/in0_V] [get_bd_intf_pins MatrixVectorActivation_3/MatrixVectorActivation_3/in0_V]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_3/out_V] [get_bd_intf_pins MatrixVectorActivation_3/MatrixVectorActivation_3/out_V]
save_bd_design
make_bd_pins_external [get_bd_pins StreamingFIFO_0/ap_clk]
set_property name ap_clk [get_bd_ports ap_clk_0]
make_bd_pins_external [get_bd_pins StreamingFIFO_0/ap_rst_n]
set_property name ap_rst_n [get_bd_ports ap_rst_n_0]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins FMPadding_Batch_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins FMPadding_Batch_0/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_0/out_V] [get_bd_intf_pins FMPadding_Batch_0/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_1/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_1/ap_clk]
connect_bd_intf_net [get_bd_intf_pins FMPadding_Batch_0/out_V] [get_bd_intf_pins StreamingFIFO_1/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ConvolutionInputGenerator_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ConvolutionInputGenerator_0/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_1/out_V] [get_bd_intf_pins ConvolutionInputGenerator_0/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingDataWidthConverter_Batch_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingDataWidthConverter_Batch_0/ap_clk]
connect_bd_intf_net [get_bd_intf_pins ConvolutionInputGenerator_0/out_V] [get_bd_intf_pins StreamingDataWidthConverter_Batch_0/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_3/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_3/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingDataWidthConverter_Batch_0/out_V] [get_bd_intf_pins StreamingFIFO_3/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins MatrixVectorActivation_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins MatrixVectorActivation_0/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_3/out_V] [get_bd_intf_pins MatrixVectorActivation_0/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins Thresholding_Batch_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins Thresholding_Batch_0/ap_clk]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_0/out_V] [get_bd_intf_pins Thresholding_Batch_0/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingDataWidthConverter_Batch_1/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingDataWidthConverter_Batch_1/ap_clk]
connect_bd_intf_net [get_bd_intf_pins Thresholding_Batch_0/out_V] [get_bd_intf_pins StreamingDataWidthConverter_Batch_1/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_6/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_6/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingDataWidthConverter_Batch_1/out_V] [get_bd_intf_pins StreamingFIFO_6/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins FMPadding_Batch_1/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins FMPadding_Batch_1/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_6/out_V] [get_bd_intf_pins FMPadding_Batch_1/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingDataWidthConverter_Batch_2/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingDataWidthConverter_Batch_2/ap_clk]
connect_bd_intf_net [get_bd_intf_pins FMPadding_Batch_1/out_V] [get_bd_intf_pins StreamingDataWidthConverter_Batch_2/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_8/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_8/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingDataWidthConverter_Batch_2/out_V] [get_bd_intf_pins StreamingFIFO_8/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ConvolutionInputGenerator_1/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ConvolutionInputGenerator_1/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_8/out_V] [get_bd_intf_pins ConvolutionInputGenerator_1/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingDataWidthConverter_Batch_3/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingDataWidthConverter_Batch_3/ap_clk]
connect_bd_intf_net [get_bd_intf_pins ConvolutionInputGenerator_1/out_V] [get_bd_intf_pins StreamingDataWidthConverter_Batch_3/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins MatrixVectorActivation_1/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins MatrixVectorActivation_1/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingDataWidthConverter_Batch_3/out_V] [get_bd_intf_pins MatrixVectorActivation_1/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins Thresholding_Batch_1/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins Thresholding_Batch_1/ap_clk]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_1/out_V] [get_bd_intf_pins Thresholding_Batch_1/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_12/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_12/ap_clk]
connect_bd_intf_net [get_bd_intf_pins Thresholding_Batch_1/out_V] [get_bd_intf_pins StreamingFIFO_12/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ConvolutionInputGenerator_2/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ConvolutionInputGenerator_2/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_12/out_V] [get_bd_intf_pins ConvolutionInputGenerator_2/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins Pool_Batch_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins Pool_Batch_0/ap_clk]
connect_bd_intf_net [get_bd_intf_pins ConvolutionInputGenerator_2/out_V] [get_bd_intf_pins Pool_Batch_0/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingDataWidthConverter_Batch_4/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingDataWidthConverter_Batch_4/ap_clk]
connect_bd_intf_net [get_bd_intf_pins Pool_Batch_0/out_V] [get_bd_intf_pins StreamingDataWidthConverter_Batch_4/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_15/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_15/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingDataWidthConverter_Batch_4/out_V] [get_bd_intf_pins StreamingFIFO_15/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins FMPadding_Batch_2/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins FMPadding_Batch_2/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_15/out_V] [get_bd_intf_pins FMPadding_Batch_2/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingDataWidthConverter_Batch_5/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingDataWidthConverter_Batch_5/ap_clk]
connect_bd_intf_net [get_bd_intf_pins FMPadding_Batch_2/out_V] [get_bd_intf_pins StreamingDataWidthConverter_Batch_5/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_17/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_17/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingDataWidthConverter_Batch_5/out_V] [get_bd_intf_pins StreamingFIFO_17/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ConvolutionInputGenerator_3/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ConvolutionInputGenerator_3/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_17/out_V] [get_bd_intf_pins ConvolutionInputGenerator_3/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingDataWidthConverter_Batch_6/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingDataWidthConverter_Batch_6/ap_clk]
connect_bd_intf_net [get_bd_intf_pins ConvolutionInputGenerator_3/out_V] [get_bd_intf_pins StreamingDataWidthConverter_Batch_6/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins MatrixVectorActivation_2/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins MatrixVectorActivation_2/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingDataWidthConverter_Batch_6/out_V] [get_bd_intf_pins MatrixVectorActivation_2/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins Thresholding_Batch_2/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins Thresholding_Batch_2/ap_clk]
connect_bd_intf_net [get_bd_intf_pins MatrixVectorActivation_2/out_V] [get_bd_intf_pins Thresholding_Batch_2/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingDataWidthConverter_Batch_7/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingDataWidthConverter_Batch_7/ap_clk]
connect_bd_intf_net [get_bd_intf_pins Thresholding_Batch_2/out_V] [get_bd_intf_pins StreamingDataWidthConverter_Batch_7/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_22/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_22/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingDataWidthConverter_Batch_7/out_V] [get_bd_intf_pins StreamingFIFO_22/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins FMPadding_Batch_3/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins FMPadding_Batch_3/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_22/out_V] [get_bd_intf_pins FMPadding_Batch_3/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_23/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_23/ap_clk]
connect_bd_intf_net [get_bd_intf_pins FMPadding_Batch_3/out_V] [get_bd_intf_pins StreamingFIFO_23/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ConvolutionInputGenerator_4/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ConvolutionInputGenerator_4/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_23/out_V] [get_bd_intf_pins ConvolutionInputGenerator_4/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins Pool_Batch_1/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins Pool_Batch_1/ap_clk]
connect_bd_intf_net [get_bd_intf_pins ConvolutionInputGenerator_4/out_V] [get_bd_intf_pins Pool_Batch_1/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingDataWidthConverter_Batch_8/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingDataWidthConverter_Batch_8/ap_clk]
connect_bd_intf_net [get_bd_intf_pins Pool_Batch_1/out_V] [get_bd_intf_pins StreamingDataWidthConverter_Batch_8/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins StreamingFIFO_26/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins StreamingFIFO_26/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingDataWidthConverter_Batch_8/out_V] [get_bd_intf_pins StreamingFIFO_26/in0_V]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins MatrixVectorActivation_3/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins MatrixVectorActivation_3/ap_clk]
connect_bd_intf_net [get_bd_intf_pins StreamingFIFO_26/out_V] [get_bd_intf_pins MatrixVectorActivation_3/in0_V]
make_bd_intf_pins_external [get_bd_intf_pins StreamingFIFO_0/in0_V]
set_property name s_axis_0 [get_bd_intf_ports in0_V_0]
make_bd_intf_pins_external [get_bd_intf_pins MatrixVectorActivation_3/out_V]
set_property name m_axis_0 [get_bd_intf_ports out_V_0]
set_property CONFIG.FREQ_HZ 100000000.000000 [get_bd_ports /ap_clk]
regenerate_bd_layout
validate_bd_design
save_bd_design
make_wrapper -files [get_files /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/vivado_stitch_proj_embwhftp/finn_vivado_stitch_proj.srcs/sources_1/bd/finn_design/finn_design.bd] -top
add_files -norecurse /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/vivado_stitch_proj_embwhftp/finn_vivado_stitch_proj.srcs/sources_1/bd/finn_design/hdl/finn_design_wrapper.v
ipx::package_project -root_dir /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/vivado_stitch_proj_embwhftp/ip -vendor xilinx_finn -library finn -taxonomy /UserIP -module finn_design -import_files
ipx::remove_segment -quiet m_axi_gmem0:APERTURE_0 [ipx::get_address_spaces m_axi_gmem0 -of_objects [ipx::current_core]]
set_property core_revision 2 [ipx::find_open_core xilinx_finn:finn:finn_design:1.0]
ipx::create_xgui_files [ipx::find_open_core xilinx_finn:finn:finn_design:1.0]
set_property value_resolve_type user [ipx::get_bus_parameters -of [ipx::get_bus_interfaces -of [ipx::current_core ]]]
file copy -force data ip/
ipx::add_file_group -type software_driver {} [ipx::current_core]
set_property type mdd [ipx::add_file data/finn_design.mdd [ipx::get_file_groups xilinx_softwaredriver -of_objects [ipx::current_core]]]
set_property type tclSource [ipx::add_file data/finn_design.tcl [ipx::get_file_groups xilinx_softwaredriver -of_objects [ipx::current_core]]]
ipx::update_checksums [ipx::find_open_core xilinx_finn:finn:finn_design:1.0]
ipx::save_core [ipx::find_open_core xilinx_finn:finn:finn_design:1.0]
set all_v_files [get_files -filter {USED_IN_SYNTHESIS == 1 && (FILE_TYPE == Verilog || FILE_TYPE == SystemVerilog || FILE_TYPE =="Verilog Header")}]
set fp [open /scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/vivado_stitch_proj_embwhftp/all_verilog_srcs.txt w]
foreach vf $all_v_files {puts $fp $vf}
close $fp
