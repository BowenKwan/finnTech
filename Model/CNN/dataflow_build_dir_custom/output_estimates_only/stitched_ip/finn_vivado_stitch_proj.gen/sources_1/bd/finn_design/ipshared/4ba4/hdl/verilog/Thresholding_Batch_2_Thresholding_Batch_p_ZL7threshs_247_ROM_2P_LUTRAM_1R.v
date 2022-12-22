// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
(* rom_style = "distributed" *) module Thresholding_Batch_2_Thresholding_Batch_p_ZL7threshs_247_ROM_2P_LUTRAM_1R (
address0, ce0, q0, reset,clk);

parameter DataWidth = 20;
parameter AddressWidth = 6;
parameter AddressRange = 64;

input[AddressWidth-1:0] address0;
input ce0;
output reg[DataWidth-1:0] q0;
input reset;
input clk;

(* ram_style = "distributed" *)reg [DataWidth-1:0] ram[0:AddressRange-1];

initial begin
    $readmemh("/scratch/bkwan/FINN/finn/ML_LL/Model/CNN-1D/tmp/code_gen_ipgen_Thresholding_Batch_2__scv1oh_/project_Thresholding_Batch_2/sol1/impl/ip/hdl/verilog/Thresholding_Batch_2_Thresholding_Batch_p_ZL7threshs_247_ROM_2P_LUTRAM_1R.dat", ram);
end



always @(posedge clk)  
begin 
    if (ce0) 
    begin
        q0 <= ram[address0];
    end
end



endmodule

