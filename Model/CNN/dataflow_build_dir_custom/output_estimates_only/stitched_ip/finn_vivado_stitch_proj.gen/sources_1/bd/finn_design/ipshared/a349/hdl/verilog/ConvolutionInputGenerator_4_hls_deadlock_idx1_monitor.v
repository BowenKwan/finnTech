`timescale 1 ns / 1 ps

module ConvolutionInputGenerator_4_hls_deadlock_idx1_monitor ( // for module ConvolutionInputGenerator_4_ConvolutionInputGenerator_4_inst.grp_ConvolutionInputGenerator_dws_2u_64u_8u_5u_2u_2u_2u_ap_resource_lutram_s_fu_28
    input wire clock,
    input wire reset,
    input wire [1:0] axis_block_sigs,
    input wire [1:0] inst_idle_sigs,
    input wire [0:0] inst_block_sigs,
    output wire block
);

// signal declare
reg monitor_find_block;
wire sub_parallel_block;
wire all_sub_parallel_has_block;
wire all_sub_single_has_block;
wire cur_axis_has_block;
wire seq_is_axis_block;

assign block = monitor_find_block;
assign all_sub_parallel_has_block = 1'b0;
assign all_sub_single_has_block = 1'b0;
assign cur_axis_has_block = 1'b0 | axis_block_sigs[0] | axis_block_sigs[1];
assign seq_is_axis_block = all_sub_parallel_has_block | all_sub_single_has_block | cur_axis_has_block;

always @(posedge clock) begin
    if (reset == 1'b1)
        monitor_find_block <= 1'b0;
    else if (seq_is_axis_block == 1'b1)
        monitor_find_block <= 1'b1;
    else
        monitor_find_block <= 1'b0;
end


// instant sub module
endmodule