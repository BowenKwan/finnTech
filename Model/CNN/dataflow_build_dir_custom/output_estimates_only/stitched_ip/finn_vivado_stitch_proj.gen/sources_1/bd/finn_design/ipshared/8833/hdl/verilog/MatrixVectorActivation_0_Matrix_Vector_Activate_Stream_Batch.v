// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module MatrixVectorActivation_0_Matrix_Vector_Activate_Stream_Batch (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        in0_V_TVALID,
        weights_V_TVALID,
        out_V_TREADY,
        in0_V_TDATA,
        in0_V_TREADY,
        out_V_TDATA,
        out_V_TVALID,
        weights_V_TDATA,
        weights_V_TREADY
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter3_fsm_state4 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;
parameter    ap_ST_iter3_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   in0_V_TVALID;
input   weights_V_TVALID;
input   out_V_TREADY;
input  [71:0] in0_V_TDATA;
output   in0_V_TREADY;
output  [23:0] out_V_TDATA;
output   out_V_TVALID;
input  [71:0] weights_V_TDATA;
output   weights_V_TREADY;

reg ap_idle;
reg in0_V_TREADY;
reg out_V_TVALID;
reg weights_V_TREADY;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg   [1:0] ap_CS_iter3_fsm;
wire    ap_CS_iter3_fsm_state0;
wire   [0:0] icmp_ln248_fu_150_p2;
wire   [0:0] icmp_ln252_fu_165_p2;
reg    ap_predicate_op26_read_state1;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state2_pp0_stage0_iter1;
wire    ap_CS_iter1_fsm_state2;
wire    ap_block_state3_pp0_stage0_iter2;
wire    ap_CS_iter2_fsm_state3;
reg   [0:0] icmp_ln248_reg_620;
reg   [0:0] icmp_ln248_reg_620_pp0_iter2_reg;
reg    ap_block_state4_pp0_stage0_iter3;
reg    ap_block_state4_io;
wire    ap_CS_iter3_fsm_state4;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    in0_V_TDATA_blk_n;
reg    out_V_TDATA_blk_n;
reg    weights_V_TDATA_blk_n;
wire   [0:0] icmp_ln248_reg_620_pp0_iter0_reg;
reg   [0:0] icmp_ln248_reg_620_pp0_iter1_reg;
wire   [7:0] local_temp_V_fu_179_p1;
reg  signed [7:0] local_temp_V_reg_627;
reg  signed [7:0] local_temp_V_9_reg_632;
reg  signed [7:0] local_temp_V_10_reg_637;
reg  signed [7:0] local_temp_V_11_reg_642;
reg  signed [7:0] local_temp_V_12_reg_647;
reg  signed [7:0] local_temp_V_13_reg_652;
reg  signed [7:0] local_temp_V_14_reg_657;
reg  signed [7:0] local_temp_V_15_reg_662;
reg  signed [7:0] local_temp_V_8_reg_667;
wire  signed [15:0] ret_V_fu_307_p2;
reg  signed [15:0] ret_V_reg_672;
wire  signed [15:0] ret_V_1_fu_330_p2;
reg  signed [15:0] ret_V_1_reg_677;
wire  signed [15:0] ret_V_2_fu_353_p2;
reg  signed [15:0] ret_V_2_reg_682;
wire  signed [15:0] ret_V_3_fu_376_p2;
reg  signed [15:0] ret_V_3_reg_687;
wire  signed [15:0] ret_V_4_fu_399_p2;
reg  signed [15:0] ret_V_4_reg_692;
wire  signed [15:0] ret_V_5_fu_422_p2;
reg  signed [15:0] ret_V_5_reg_697;
wire  signed [15:0] ret_V_6_fu_445_p2;
reg  signed [15:0] ret_V_6_reg_702;
wire  signed [15:0] ret_V_7_fu_468_p2;
reg  signed [15:0] ret_V_7_reg_707;
wire  signed [15:0] ret_V_8_fu_491_p2;
reg  signed [15:0] ret_V_8_reg_712;
wire   [17:0] add_ln886_2_fu_544_p2;
reg   [17:0] add_ln886_2_reg_717;
wire   [17:0] add_ln886_6_fu_576_p2;
reg   [17:0] add_ln886_6_reg_722;
reg   [31:0] nf_fu_106;
wire   [31:0] nf_2_fu_275_p3;
wire    ap_loop_init;
reg   [31:0] ap_sig_allocacmp_nf_load_1;
reg   [31:0] ap_sig_allocacmp_nf_load;
reg   [9:0] i_fu_110;
wire   [9:0] i_2_fu_156_p2;
reg   [9:0] ap_sig_allocacmp_i_1;
reg   [71:0] p_Val2_s_fu_114;
wire   [31:0] nf_1_fu_263_p2;
wire   [0:0] icmp_ln301_fu_269_p2;
wire  signed [7:0] r_V_fu_296_p1;
wire  signed [7:0] r_V_1_fu_313_p4;
wire  signed [7:0] r_V_2_fu_336_p4;
wire  signed [7:0] r_V_3_fu_359_p4;
wire  signed [7:0] r_V_4_fu_382_p4;
wire  signed [7:0] r_V_5_fu_405_p4;
wire  signed [7:0] r_V_6_fu_428_p4;
wire  signed [7:0] r_V_7_fu_451_p4;
wire  signed [7:0] r_V_8_fu_474_p4;
wire  signed [16:0] sext_ln674_1_fu_500_p1;
wire  signed [16:0] sext_ln674_fu_497_p1;
wire   [16:0] add_ln886_fu_524_p2;
wire  signed [16:0] sext_ln674_2_fu_503_p1;
wire  signed [16:0] sext_ln674_3_fu_506_p1;
wire   [16:0] add_ln886_1_fu_534_p2;
wire  signed [17:0] sext_ln886_2_fu_540_p1;
wire  signed [17:0] sext_ln886_1_fu_530_p1;
wire  signed [16:0] sext_ln674_4_fu_509_p1;
wire  signed [16:0] sext_ln674_5_fu_512_p1;
wire   [16:0] add_ln886_3_fu_550_p2;
wire  signed [16:0] sext_ln674_7_fu_518_p1;
wire  signed [16:0] sext_ln886_fu_521_p1;
wire   [16:0] add_ln886_4_fu_560_p2;
wire  signed [16:0] sext_ln674_6_fu_515_p1;
wire   [16:0] add_ln886_5_fu_566_p2;
wire  signed [17:0] sext_ln886_5_fu_572_p1;
wire  signed [17:0] sext_ln886_4_fu_556_p1;
wire  signed [18:0] sext_ln886_6_fu_585_p1;
wire  signed [18:0] sext_ln886_3_fu_582_p1;
wire   [18:0] outElem_m_val_V_fu_588_p2;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg   [1:0] ap_NS_iter3_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
wire    ap_ST_iter2_fsm_state3_blk;
reg    ap_ST_iter3_fsm_state4_blk;
wire    ap_start_int;
reg    ap_condition_571;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_iter0_fsm = 1'd1;
#0 ap_CS_iter1_fsm = 2'd1;
#0 ap_CS_iter2_fsm = 2'd1;
#0 ap_CS_iter3_fsm = 2'd1;
#0 ap_done_reg = 1'b0;
end

MatrixVectorActivation_0_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U1(
    .din0(r_V_fu_296_p1),
    .din1(local_temp_V_reg_627),
    .dout(ret_V_fu_307_p2)
);

MatrixVectorActivation_0_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U2(
    .din0(r_V_1_fu_313_p4),
    .din1(local_temp_V_9_reg_632),
    .dout(ret_V_1_fu_330_p2)
);

MatrixVectorActivation_0_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U3(
    .din0(r_V_2_fu_336_p4),
    .din1(local_temp_V_10_reg_637),
    .dout(ret_V_2_fu_353_p2)
);

MatrixVectorActivation_0_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U4(
    .din0(r_V_3_fu_359_p4),
    .din1(local_temp_V_11_reg_642),
    .dout(ret_V_3_fu_376_p2)
);

MatrixVectorActivation_0_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U5(
    .din0(r_V_4_fu_382_p4),
    .din1(local_temp_V_12_reg_647),
    .dout(ret_V_4_fu_399_p2)
);

MatrixVectorActivation_0_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U6(
    .din0(r_V_5_fu_405_p4),
    .din1(local_temp_V_13_reg_652),
    .dout(ret_V_5_fu_422_p2)
);

MatrixVectorActivation_0_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U7(
    .din0(r_V_6_fu_428_p4),
    .din1(local_temp_V_14_reg_657),
    .dout(ret_V_6_fu_445_p2)
);

MatrixVectorActivation_0_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U8(
    .din0(r_V_7_fu_451_p4),
    .din1(local_temp_V_15_reg_662),
    .dout(ret_V_7_fu_468_p2)
);

MatrixVectorActivation_0_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U9(
    .din0(r_V_8_fu_474_p4),
    .din1(local_temp_V_8_reg_667),
    .dout(ret_V_8_fu_491_p2)
);

MatrixVectorActivation_0_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter3_fsm <= ap_ST_iter3_fsm_state0;
    end else begin
        ap_CS_iter3_fsm <= ap_NS_iter3_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if ((~((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0))) & (ap_loop_exit_ready_pp0_iter3_reg == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0))) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b0) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_571)) begin
        if ((icmp_ln248_fu_150_p2 == 1'd0)) begin
            i_fu_110 <= i_2_fu_156_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            i_fu_110 <= 10'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_571)) begin
        if ((icmp_ln248_fu_150_p2 == 1'd0)) begin
            nf_fu_106 <= nf_2_fu_275_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            nf_fu_106 <= 32'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & (1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln248_reg_620_pp0_iter1_reg == 1'd0))) begin
        add_ln886_2_reg_717 <= add_ln886_2_fu_544_p2;
        add_ln886_6_reg_722 <= add_ln886_6_fu_576_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_start_int == 1'b0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln248_reg_620 <= icmp_ln248_fu_150_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln248_reg_620_pp0_iter1_reg <= icmp_ln248_reg_620;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        icmp_ln248_reg_620_pp0_iter2_reg <= icmp_ln248_reg_620_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_start_int == 1'b0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (icmp_ln248_fu_150_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        local_temp_V_10_reg_637 <= {{weights_V_TDATA[23:16]}};
        local_temp_V_11_reg_642 <= {{weights_V_TDATA[31:24]}};
        local_temp_V_12_reg_647 <= {{weights_V_TDATA[39:32]}};
        local_temp_V_13_reg_652 <= {{weights_V_TDATA[47:40]}};
        local_temp_V_14_reg_657 <= {{weights_V_TDATA[55:48]}};
        local_temp_V_15_reg_662 <= {{weights_V_TDATA[63:56]}};
        local_temp_V_8_reg_667 <= {{weights_V_TDATA[71:64]}};
        local_temp_V_9_reg_632 <= {{weights_V_TDATA[15:8]}};
        local_temp_V_reg_627 <= local_temp_V_fu_179_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_start_int == 1'b0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (icmp_ln252_fu_165_p2 == 1'd1) & (icmp_ln248_fu_150_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        p_Val2_s_fu_114 <= in0_V_TDATA;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & (1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln248_reg_620_pp0_iter0_reg == 1'd0))) begin
        ret_V_1_reg_677 <= ret_V_1_fu_330_p2;
        ret_V_2_reg_682 <= ret_V_2_fu_353_p2;
        ret_V_3_reg_687 <= ret_V_3_fu_376_p2;
        ret_V_4_reg_692 <= ret_V_4_fu_399_p2;
        ret_V_5_reg_697 <= ret_V_5_fu_422_p2;
        ret_V_6_reg_702 <= ret_V_6_fu_445_p2;
        ret_V_7_reg_707 <= ret_V_7_fu_468_p2;
        ret_V_8_reg_712 <= ret_V_8_fu_491_p2;
        ret_V_reg_672 <= ret_V_fu_307_p2;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0)))) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter1_fsm_state2_blk = 1'b0;

assign ap_ST_iter2_fsm_state3_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) begin
        ap_ST_iter3_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter3_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_start_int == 1'b0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (icmp_ln248_fu_150_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0))) & (ap_loop_exit_ready_pp0_iter3_reg == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_start_int == 1'b0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_i_1 = 10'd0;
    end else begin
        ap_sig_allocacmp_i_1 = i_fu_110;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_nf_load = 32'd0;
    end else begin
        ap_sig_allocacmp_nf_load = nf_fu_106;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_nf_load_1 = 32'd0;
    end else begin
        ap_sig_allocacmp_nf_load_1 = nf_fu_106;
    end
end

always @ (*) begin
    if (((ap_predicate_op26_read_state1 == 1'b1) & (ap_start_int == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        in0_V_TDATA_blk_n = in0_V_TVALID;
    end else begin
        in0_V_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_start_int == 1'b0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (ap_predicate_op26_read_state1 == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        in0_V_TREADY = 1'b1;
    end else begin
        in0_V_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        out_V_TDATA_blk_n = out_V_TREADY;
    end else begin
        out_V_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0))) & (icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        out_V_TVALID = 1'b1;
    end else begin
        out_V_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b1) & (icmp_ln248_fu_150_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        weights_V_TDATA_blk_n = weights_V_TVALID;
    end else begin
        weights_V_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_start_int == 1'b0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (icmp_ln248_fu_150_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        weights_V_TREADY = 1'b1;
    end else begin
        weights_V_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state2 : begin
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & ~((ap_start_int == 1'b0) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & ((ap_start_int == 1'b0) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((ap_start_int == 1'b0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state3 : begin
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter3_fsm)
        ap_ST_iter3_fsm_state4 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0))) & (1'b0 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else if (((~((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0))) & (1'b1 == ap_CS_iter2_fsm_state3)) | (~((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0))) & (icmp_ln248_reg_620_pp0_iter2_reg == 1'd1) & (1'b1 == ap_CS_iter3_fsm_state4)))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter3_fsm = 'bx;
        end
    endcase
end

assign add_ln886_1_fu_534_p2 = ($signed(sext_ln674_2_fu_503_p1) + $signed(sext_ln674_3_fu_506_p1));

assign add_ln886_2_fu_544_p2 = ($signed(sext_ln886_2_fu_540_p1) + $signed(sext_ln886_1_fu_530_p1));

assign add_ln886_3_fu_550_p2 = ($signed(sext_ln674_4_fu_509_p1) + $signed(sext_ln674_5_fu_512_p1));

assign add_ln886_4_fu_560_p2 = ($signed(sext_ln674_7_fu_518_p1) + $signed(sext_ln886_fu_521_p1));

assign add_ln886_5_fu_566_p2 = ($signed(add_ln886_4_fu_560_p2) + $signed(sext_ln674_6_fu_515_p1));

assign add_ln886_6_fu_576_p2 = ($signed(sext_ln886_5_fu_572_p1) + $signed(sext_ln886_4_fu_556_p1));

assign add_ln886_fu_524_p2 = ($signed(sext_ln674_1_fu_500_p1) + $signed(sext_ln674_fu_497_p1));

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state4 = ap_CS_iter3_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0)));
end

assign ap_block_state2_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state3_pp0_stage0_iter2 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state4_io = ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state4_pp0_stage0_iter3 = ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0));
end

always @ (*) begin
    ap_condition_571 = (~((ap_start_int == 1'b0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | ((icmp_ln248_reg_620_pp0_iter2_reg == 1'd0) & (out_V_TREADY == 1'b0)))) | ((weights_V_TVALID == 1'b0) & (icmp_ln248_fu_150_p2 == 1'd0)) | ((ap_predicate_op26_read_state1 == 1'b1) & (in0_V_TVALID == 1'b0))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

always @ (*) begin
    ap_predicate_op26_read_state1 = ((icmp_ln252_fu_165_p2 == 1'd1) & (icmp_ln248_fu_150_p2 == 1'd0));
end

assign i_2_fu_156_p2 = (ap_sig_allocacmp_i_1 + 10'd1);

assign icmp_ln248_fu_150_p2 = ((ap_sig_allocacmp_i_1 == 10'd784) ? 1'b1 : 1'b0);

assign icmp_ln248_reg_620_pp0_iter0_reg = icmp_ln248_reg_620;

assign icmp_ln252_fu_165_p2 = ((ap_sig_allocacmp_nf_load_1 == 32'd0) ? 1'b1 : 1'b0);

assign icmp_ln301_fu_269_p2 = ((nf_1_fu_263_p2 == 32'd16) ? 1'b1 : 1'b0);

assign local_temp_V_fu_179_p1 = weights_V_TDATA[7:0];

assign nf_1_fu_263_p2 = (ap_sig_allocacmp_nf_load + 32'd1);

assign nf_2_fu_275_p3 = ((icmp_ln301_fu_269_p2[0:0] == 1'b1) ? 32'd0 : nf_1_fu_263_p2);

assign outElem_m_val_V_fu_588_p2 = ($signed(sext_ln886_6_fu_585_p1) + $signed(sext_ln886_3_fu_582_p1));

assign out_V_TDATA = $signed(outElem_m_val_V_fu_588_p2);

assign r_V_1_fu_313_p4 = {{p_Val2_s_fu_114[15:8]}};

assign r_V_2_fu_336_p4 = {{p_Val2_s_fu_114[23:16]}};

assign r_V_3_fu_359_p4 = {{p_Val2_s_fu_114[31:24]}};

assign r_V_4_fu_382_p4 = {{p_Val2_s_fu_114[39:32]}};

assign r_V_5_fu_405_p4 = {{p_Val2_s_fu_114[47:40]}};

assign r_V_6_fu_428_p4 = {{p_Val2_s_fu_114[55:48]}};

assign r_V_7_fu_451_p4 = {{p_Val2_s_fu_114[63:56]}};

assign r_V_8_fu_474_p4 = {{p_Val2_s_fu_114[71:64]}};

assign r_V_fu_296_p1 = p_Val2_s_fu_114[7:0];

assign sext_ln674_1_fu_500_p1 = ret_V_1_reg_677;

assign sext_ln674_2_fu_503_p1 = ret_V_2_reg_682;

assign sext_ln674_3_fu_506_p1 = ret_V_3_reg_687;

assign sext_ln674_4_fu_509_p1 = ret_V_4_reg_692;

assign sext_ln674_5_fu_512_p1 = ret_V_5_reg_697;

assign sext_ln674_6_fu_515_p1 = ret_V_6_reg_702;

assign sext_ln674_7_fu_518_p1 = ret_V_7_reg_707;

assign sext_ln674_fu_497_p1 = ret_V_reg_672;

assign sext_ln886_1_fu_530_p1 = $signed(add_ln886_fu_524_p2);

assign sext_ln886_2_fu_540_p1 = $signed(add_ln886_1_fu_534_p2);

assign sext_ln886_3_fu_582_p1 = $signed(add_ln886_2_reg_717);

assign sext_ln886_4_fu_556_p1 = $signed(add_ln886_3_fu_550_p2);

assign sext_ln886_5_fu_572_p1 = $signed(add_ln886_5_fu_566_p2);

assign sext_ln886_6_fu_585_p1 = $signed(add_ln886_6_reg_722);

assign sext_ln886_fu_521_p1 = ret_V_8_reg_712;

endmodule //MatrixVectorActivation_0_Matrix_Vector_Activate_Stream_Batch
