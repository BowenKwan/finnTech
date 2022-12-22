//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2022.1 (lin64) Build 3526262 Mon Apr 18 15:47:01 MDT 2022
//Date        : Tue Nov 15 11:51:09 2022
//Host        : finn_dev_pokyeek running 64-bit Ubuntu 18.04.5 LTS
//Command     : generate_target finn_design.bd
//Design      : finn_design
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module MatrixVectorActivation_0_imp_MT0NP
   (ap_clk,
    ap_rst_n,
    in0_V_tdata,
    in0_V_tready,
    in0_V_tvalid,
    out_V_tdata,
    out_V_tready,
    out_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [71:0]in0_V_tdata;
  output in0_V_tready;
  input in0_V_tvalid;
  output [23:0]out_V_tdata;
  input out_V_tready;
  output out_V_tvalid;

  wire [23:0]MatrixVectorActivation_0_out_V_TDATA;
  wire MatrixVectorActivation_0_out_V_TREADY;
  wire MatrixVectorActivation_0_out_V_TVALID;
  wire [71:0]MatrixVectorActivation_0_wstrm_m_axis_0_TDATA;
  wire MatrixVectorActivation_0_wstrm_m_axis_0_TREADY;
  wire MatrixVectorActivation_0_wstrm_m_axis_0_TVALID;
  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [71:0]in0_V_1_TDATA;
  wire in0_V_1_TREADY;
  wire in0_V_1_TVALID;

  assign MatrixVectorActivation_0_out_V_TREADY = out_V_tready;
  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign in0_V_1_TDATA = in0_V_tdata[71:0];
  assign in0_V_1_TVALID = in0_V_tvalid;
  assign in0_V_tready = in0_V_1_TREADY;
  assign out_V_tdata[23:0] = MatrixVectorActivation_0_out_V_TDATA;
  assign out_V_tvalid = MatrixVectorActivation_0_out_V_TVALID;
  finn_design_MatrixVectorActivation_0_0 MatrixVectorActivation_0
       (.ap_clk(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .in0_V_TDATA(in0_V_1_TDATA),
        .in0_V_TREADY(in0_V_1_TREADY),
        .in0_V_TVALID(in0_V_1_TVALID),
        .out_V_TDATA(MatrixVectorActivation_0_out_V_TDATA),
        .out_V_TREADY(MatrixVectorActivation_0_out_V_TREADY),
        .out_V_TVALID(MatrixVectorActivation_0_out_V_TVALID),
        .weights_V_TDATA(MatrixVectorActivation_0_wstrm_m_axis_0_TDATA),
        .weights_V_TREADY(MatrixVectorActivation_0_wstrm_m_axis_0_TREADY),
        .weights_V_TVALID(MatrixVectorActivation_0_wstrm_m_axis_0_TVALID));
  finn_design_MatrixVectorActivation_0_wstrm_0 MatrixVectorActivation_0_wstrm
       (.aclk(ap_clk_1),
        .araddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .aresetn(ap_rst_n_1),
        .arprot({1'b0,1'b0,1'b0}),
        .arvalid(1'b0),
        .awaddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .awprot({1'b0,1'b0,1'b0}),
        .awvalid(1'b0),
        .bready(1'b0),
        .m_axis_0_tdata(MatrixVectorActivation_0_wstrm_m_axis_0_TDATA),
        .m_axis_0_tready(MatrixVectorActivation_0_wstrm_m_axis_0_TREADY),
        .m_axis_0_tvalid(MatrixVectorActivation_0_wstrm_m_axis_0_TVALID),
        .rready(1'b0),
        .wdata({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .wstrb({1'b1,1'b1,1'b1,1'b1}),
        .wvalid(1'b0));
endmodule

module MatrixVectorActivation_1_imp_16PJ27U
   (ap_clk,
    ap_rst_n,
    in0_V_tdata,
    in0_V_tready,
    in0_V_tvalid,
    out_V_tdata,
    out_V_tready,
    out_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [1151:0]in0_V_tdata;
  output in0_V_tready;
  input in0_V_tvalid;
  output [47:0]out_V_tdata;
  input out_V_tready;
  output out_V_tvalid;

  wire [47:0]MatrixVectorActivation_1_out_V_TDATA;
  wire MatrixVectorActivation_1_out_V_TREADY;
  wire MatrixVectorActivation_1_out_V_TVALID;
  wire [2303:0]MatrixVectorActivation_1_wstrm_m_axis_0_TDATA;
  wire MatrixVectorActivation_1_wstrm_m_axis_0_TREADY;
  wire MatrixVectorActivation_1_wstrm_m_axis_0_TVALID;
  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [1151:0]in0_V_1_TDATA;
  wire in0_V_1_TREADY;
  wire in0_V_1_TVALID;

  assign MatrixVectorActivation_1_out_V_TREADY = out_V_tready;
  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign in0_V_1_TDATA = in0_V_tdata[1151:0];
  assign in0_V_1_TVALID = in0_V_tvalid;
  assign in0_V_tready = in0_V_1_TREADY;
  assign out_V_tdata[47:0] = MatrixVectorActivation_1_out_V_TDATA;
  assign out_V_tvalid = MatrixVectorActivation_1_out_V_TVALID;
  finn_design_MatrixVectorActivation_1_0 MatrixVectorActivation_1
       (.ap_clk(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .in0_V_TDATA(in0_V_1_TDATA),
        .in0_V_TREADY(in0_V_1_TREADY),
        .in0_V_TVALID(in0_V_1_TVALID),
        .out_V_TDATA(MatrixVectorActivation_1_out_V_TDATA),
        .out_V_TREADY(MatrixVectorActivation_1_out_V_TREADY),
        .out_V_TVALID(MatrixVectorActivation_1_out_V_TVALID),
        .weights_V_TDATA(MatrixVectorActivation_1_wstrm_m_axis_0_TDATA),
        .weights_V_TREADY(MatrixVectorActivation_1_wstrm_m_axis_0_TREADY),
        .weights_V_TVALID(MatrixVectorActivation_1_wstrm_m_axis_0_TVALID));
  finn_design_MatrixVectorActivation_1_wstrm_0 MatrixVectorActivation_1_wstrm
       (.aclk(ap_clk_1),
        .araddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .aresetn(ap_rst_n_1),
        .arprot({1'b0,1'b0,1'b0}),
        .arvalid(1'b0),
        .awaddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .awprot({1'b0,1'b0,1'b0}),
        .awvalid(1'b0),
        .bready(1'b0),
        .m_axis_0_tdata(MatrixVectorActivation_1_wstrm_m_axis_0_TDATA),
        .m_axis_0_tready(MatrixVectorActivation_1_wstrm_m_axis_0_TREADY),
        .m_axis_0_tvalid(MatrixVectorActivation_1_wstrm_m_axis_0_TVALID),
        .rready(1'b0),
        .wdata({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .wstrb({1'b1,1'b1,1'b1,1'b1}),
        .wvalid(1'b0));
endmodule

module MatrixVectorActivation_2_imp_1U5U4NE
   (ap_clk,
    ap_rst_n,
    in0_V_tdata,
    in0_V_tready,
    in0_V_tvalid,
    out_V_tdata,
    out_V_tready,
    out_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [2303:0]in0_V_tdata;
  output in0_V_tready;
  input in0_V_tvalid;
  output [31:0]out_V_tdata;
  input out_V_tready;
  output out_V_tvalid;

  wire [31:0]MatrixVectorActivation_2_out_V_TDATA;
  wire MatrixVectorActivation_2_out_V_TREADY;
  wire MatrixVectorActivation_2_out_V_TVALID;
  wire [2303:0]MatrixVectorActivation_2_wstrm_m_axis_0_TDATA;
  wire MatrixVectorActivation_2_wstrm_m_axis_0_TREADY;
  wire MatrixVectorActivation_2_wstrm_m_axis_0_TVALID;
  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [2303:0]in0_V_1_TDATA;
  wire in0_V_1_TREADY;
  wire in0_V_1_TVALID;

  assign MatrixVectorActivation_2_out_V_TREADY = out_V_tready;
  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign in0_V_1_TDATA = in0_V_tdata[2303:0];
  assign in0_V_1_TVALID = in0_V_tvalid;
  assign in0_V_tready = in0_V_1_TREADY;
  assign out_V_tdata[31:0] = MatrixVectorActivation_2_out_V_TDATA;
  assign out_V_tvalid = MatrixVectorActivation_2_out_V_TVALID;
  finn_design_MatrixVectorActivation_2_0 MatrixVectorActivation_2
       (.ap_clk(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .in0_V_TDATA(in0_V_1_TDATA),
        .in0_V_TREADY(in0_V_1_TREADY),
        .in0_V_TVALID(in0_V_1_TVALID),
        .out_V_TDATA(MatrixVectorActivation_2_out_V_TDATA),
        .out_V_TREADY(MatrixVectorActivation_2_out_V_TREADY),
        .out_V_TVALID(MatrixVectorActivation_2_out_V_TVALID),
        .weights_V_TDATA(MatrixVectorActivation_2_wstrm_m_axis_0_TDATA),
        .weights_V_TREADY(MatrixVectorActivation_2_wstrm_m_axis_0_TREADY),
        .weights_V_TVALID(MatrixVectorActivation_2_wstrm_m_axis_0_TVALID));
  finn_design_MatrixVectorActivation_2_wstrm_0 MatrixVectorActivation_2_wstrm
       (.aclk(ap_clk_1),
        .araddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .aresetn(ap_rst_n_1),
        .arprot({1'b0,1'b0,1'b0}),
        .arvalid(1'b0),
        .awaddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .awprot({1'b0,1'b0,1'b0}),
        .awvalid(1'b0),
        .bready(1'b0),
        .m_axis_0_tdata(MatrixVectorActivation_2_wstrm_m_axis_0_TDATA),
        .m_axis_0_tready(MatrixVectorActivation_2_wstrm_m_axis_0_TREADY),
        .m_axis_0_tvalid(MatrixVectorActivation_2_wstrm_m_axis_0_TVALID),
        .rready(1'b0),
        .wdata({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .wstrb({1'b1,1'b1,1'b1,1'b1}),
        .wvalid(1'b0));
endmodule

module MatrixVectorActivation_3_imp_WP21D1
   (ap_clk,
    ap_rst_n,
    in0_V_tdata,
    in0_V_tready,
    in0_V_tvalid,
    out_V_tdata,
    out_V_tready,
    out_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [31:0]in0_V_tdata;
  output in0_V_tready;
  input in0_V_tvalid;
  output [23:0]out_V_tdata;
  input out_V_tready;
  output out_V_tvalid;

  wire [23:0]MatrixVectorActivation_3_out_V_TDATA;
  wire MatrixVectorActivation_3_out_V_TREADY;
  wire MatrixVectorActivation_3_out_V_TVALID;
  wire [31:0]MatrixVectorActivation_3_wstrm_m_axis_0_TDATA;
  wire MatrixVectorActivation_3_wstrm_m_axis_0_TREADY;
  wire MatrixVectorActivation_3_wstrm_m_axis_0_TVALID;
  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [31:0]in0_V_1_TDATA;
  wire in0_V_1_TREADY;
  wire in0_V_1_TVALID;

  assign MatrixVectorActivation_3_out_V_TREADY = out_V_tready;
  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign in0_V_1_TDATA = in0_V_tdata[31:0];
  assign in0_V_1_TVALID = in0_V_tvalid;
  assign in0_V_tready = in0_V_1_TREADY;
  assign out_V_tdata[23:0] = MatrixVectorActivation_3_out_V_TDATA;
  assign out_V_tvalid = MatrixVectorActivation_3_out_V_TVALID;
  finn_design_MatrixVectorActivation_3_0 MatrixVectorActivation_3
       (.ap_clk(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .in0_V_TDATA(in0_V_1_TDATA),
        .in0_V_TREADY(in0_V_1_TREADY),
        .in0_V_TVALID(in0_V_1_TVALID),
        .out_V_TDATA(MatrixVectorActivation_3_out_V_TDATA),
        .out_V_TREADY(MatrixVectorActivation_3_out_V_TREADY),
        .out_V_TVALID(MatrixVectorActivation_3_out_V_TVALID),
        .weights_V_TDATA(MatrixVectorActivation_3_wstrm_m_axis_0_TDATA),
        .weights_V_TREADY(MatrixVectorActivation_3_wstrm_m_axis_0_TREADY),
        .weights_V_TVALID(MatrixVectorActivation_3_wstrm_m_axis_0_TVALID));
  finn_design_MatrixVectorActivation_3_wstrm_0 MatrixVectorActivation_3_wstrm
       (.aclk(ap_clk_1),
        .araddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .aresetn(ap_rst_n_1),
        .arprot({1'b0,1'b0,1'b0}),
        .arvalid(1'b0),
        .awaddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .awprot({1'b0,1'b0,1'b0}),
        .awvalid(1'b0),
        .bready(1'b0),
        .m_axis_0_tdata(MatrixVectorActivation_3_wstrm_m_axis_0_TDATA),
        .m_axis_0_tready(MatrixVectorActivation_3_wstrm_m_axis_0_TREADY),
        .m_axis_0_tvalid(MatrixVectorActivation_3_wstrm_m_axis_0_TVALID),
        .rready(1'b0),
        .wdata({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .wstrb({1'b1,1'b1,1'b1,1'b1}),
        .wvalid(1'b0));
endmodule

module StreamingFIFO_12_imp_Z91EFF
   (ap_clk,
    ap_rst_n,
    in0_V_tdata,
    in0_V_tready,
    in0_V_tvalid,
    out_V_tdata,
    out_V_tready,
    out_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [15:0]in0_V_tdata;
  output in0_V_tready;
  input in0_V_tvalid;
  output [15:0]out_V_tdata;
  input out_V_tready;
  output out_V_tvalid;

  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [15:0]fifo_M_AXIS_TDATA;
  wire fifo_M_AXIS_TREADY;
  wire fifo_M_AXIS_TVALID;
  wire [15:0]in0_V_1_TDATA;
  wire in0_V_1_TREADY;
  wire in0_V_1_TVALID;

  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign fifo_M_AXIS_TREADY = out_V_tready;
  assign in0_V_1_TDATA = in0_V_tdata[15:0];
  assign in0_V_1_TVALID = in0_V_tvalid;
  assign in0_V_tready = in0_V_1_TREADY;
  assign out_V_tdata[15:0] = fifo_M_AXIS_TDATA;
  assign out_V_tvalid = fifo_M_AXIS_TVALID;
  finn_design_fifo_0 fifo
       (.m_axis_tdata(fifo_M_AXIS_TDATA),
        .m_axis_tready(fifo_M_AXIS_TREADY),
        .m_axis_tvalid(fifo_M_AXIS_TVALID),
        .s_axis_aclk(ap_clk_1),
        .s_axis_aresetn(ap_rst_n_1),
        .s_axis_tdata(in0_V_1_TDATA),
        .s_axis_tready(in0_V_1_TREADY),
        .s_axis_tvalid(in0_V_1_TVALID));
endmodule

module StreamingFIFO_23_imp_OSV7QI
   (ap_clk,
    ap_rst_n,
    in0_V_tdata,
    in0_V_tready,
    in0_V_tvalid,
    out_V_tdata,
    out_V_tready,
    out_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [15:0]in0_V_tdata;
  output in0_V_tready;
  input in0_V_tvalid;
  output [15:0]out_V_tdata;
  input out_V_tready;
  output out_V_tvalid;

  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [15:0]fifo_M_AXIS_TDATA;
  wire fifo_M_AXIS_TREADY;
  wire fifo_M_AXIS_TVALID;
  wire [15:0]in0_V_1_TDATA;
  wire in0_V_1_TREADY;
  wire in0_V_1_TVALID;

  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign fifo_M_AXIS_TREADY = out_V_tready;
  assign in0_V_1_TDATA = in0_V_tdata[15:0];
  assign in0_V_1_TVALID = in0_V_tvalid;
  assign in0_V_tready = in0_V_1_TREADY;
  assign out_V_tdata[15:0] = fifo_M_AXIS_TDATA;
  assign out_V_tvalid = fifo_M_AXIS_TVALID;
  finn_design_fifo_1 fifo
       (.m_axis_tdata(fifo_M_AXIS_TDATA),
        .m_axis_tready(fifo_M_AXIS_TREADY),
        .m_axis_tvalid(fifo_M_AXIS_TVALID),
        .s_axis_aclk(ap_clk_1),
        .s_axis_aresetn(ap_rst_n_1),
        .s_axis_tdata(in0_V_1_TDATA),
        .s_axis_tready(in0_V_1_TREADY),
        .s_axis_tvalid(in0_V_1_TVALID));
endmodule

(* CORE_GENERATION_INFO = "finn_design,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=finn_design,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=48,numReposBlks=42,numNonXlnxBlks=0,numHierBlks=6,maxHierDepth=1,numSysgenBlks=0,numHlsBlks=27,numHdlrefBlks=0,numPkgbdBlks=0,bdsource=USER,synth_mode=OOC_per_IP}" *) (* HW_HANDOFF = "finn_design.hwdef" *) 
module finn_design
   (ap_clk,
    ap_rst_n,
    m_axis_0_tdata,
    m_axis_0_tready,
    m_axis_0_tvalid,
    s_axis_0_tdata,
    s_axis_0_tready,
    s_axis_0_tvalid);
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 CLK.AP_CLK CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME CLK.AP_CLK, ASSOCIATED_BUSIF s_axis_0:m_axis_0, ASSOCIATED_RESET ap_rst_n, CLK_DOMAIN finn_design_ap_clk_0, FREQ_HZ 100000000.000000, FREQ_TOLERANCE_HZ 0, INSERT_VIP 0, PHASE 0.0" *) input ap_clk;
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 RST.AP_RST_N RST" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME RST.AP_RST_N, INSERT_VIP 0, POLARITY ACTIVE_LOW" *) input ap_rst_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis_0 " *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME m_axis_0, CLK_DOMAIN finn_design_ap_clk_0, FREQ_HZ 100000000.000000, HAS_TKEEP 0, HAS_TLAST 0, HAS_TREADY 1, HAS_TSTRB 0, INSERT_VIP 0, LAYERED_METADATA undef, PHASE 0.0, TDATA_NUM_BYTES 3, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0" *) output [23:0]m_axis_0_tdata;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis_0 " *) input m_axis_0_tready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis_0 " *) output m_axis_0_tvalid;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis_0 " *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axis_0, CLK_DOMAIN finn_design_ap_clk_0, FREQ_HZ 100000000.000000, HAS_TKEEP 0, HAS_TLAST 0, HAS_TREADY 1, HAS_TSTRB 0, INSERT_VIP 0, LAYERED_METADATA undef, PHASE 0.0, TDATA_NUM_BYTES 1, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0" *) input [7:0]s_axis_0_tdata;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis_0 " *) output s_axis_0_tready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis_0 " *) input s_axis_0_tvalid;

  wire [7:0]ConvolutionInputGenerator_0_out_V_TDATA;
  wire ConvolutionInputGenerator_0_out_V_TREADY;
  wire ConvolutionInputGenerator_0_out_V_TVALID;
  wire [63:0]ConvolutionInputGenerator_1_out_V_TDATA;
  wire ConvolutionInputGenerator_1_out_V_TREADY;
  wire ConvolutionInputGenerator_1_out_V_TVALID;
  wire [15:0]ConvolutionInputGenerator_2_out_V_TDATA;
  wire ConvolutionInputGenerator_2_out_V_TREADY;
  wire ConvolutionInputGenerator_2_out_V_TVALID;
  wire [31:0]ConvolutionInputGenerator_3_out_V_TDATA;
  wire ConvolutionInputGenerator_3_out_V_TREADY;
  wire ConvolutionInputGenerator_3_out_V_TVALID;
  wire [15:0]ConvolutionInputGenerator_4_out_V_TDATA;
  wire ConvolutionInputGenerator_4_out_V_TREADY;
  wire ConvolutionInputGenerator_4_out_V_TVALID;
  wire [7:0]FMPadding_Batch_0_out_V_TDATA;
  wire FMPadding_Batch_0_out_V_TREADY;
  wire FMPadding_Batch_0_out_V_TVALID;
  wire [15:0]FMPadding_Batch_1_out_V_TDATA;
  wire FMPadding_Batch_1_out_V_TREADY;
  wire FMPadding_Batch_1_out_V_TVALID;
  wire [7:0]FMPadding_Batch_2_out_V_TDATA;
  wire FMPadding_Batch_2_out_V_TREADY;
  wire FMPadding_Batch_2_out_V_TVALID;
  wire [15:0]FMPadding_Batch_3_out_V_TDATA;
  wire FMPadding_Batch_3_out_V_TREADY;
  wire FMPadding_Batch_3_out_V_TVALID;
  wire [23:0]MatrixVectorActivation_0_out_V_TDATA;
  wire MatrixVectorActivation_0_out_V_TREADY;
  wire MatrixVectorActivation_0_out_V_TVALID;
  wire [47:0]MatrixVectorActivation_1_out_V_TDATA;
  wire MatrixVectorActivation_1_out_V_TREADY;
  wire MatrixVectorActivation_1_out_V_TVALID;
  wire [31:0]MatrixVectorActivation_2_out_V_TDATA;
  wire MatrixVectorActivation_2_out_V_TREADY;
  wire MatrixVectorActivation_2_out_V_TVALID;
  wire [23:0]MatrixVectorActivation_3_out_V_TDATA;
  wire MatrixVectorActivation_3_out_V_TREADY;
  wire MatrixVectorActivation_3_out_V_TVALID;
  wire [15:0]Pool_Batch_0_out_V_TDATA;
  wire Pool_Batch_0_out_V_TREADY;
  wire Pool_Batch_0_out_V_TVALID;
  wire [15:0]Pool_Batch_1_out_V_TDATA;
  wire Pool_Batch_1_out_V_TREADY;
  wire Pool_Batch_1_out_V_TVALID;
  wire [71:0]StreamingDataWidthConverter_Batch_0_out_V_TDATA;
  wire StreamingDataWidthConverter_Batch_0_out_V_TREADY;
  wire StreamingDataWidthConverter_Batch_0_out_V_TVALID;
  wire [15:0]StreamingDataWidthConverter_Batch_1_out_V_TDATA;
  wire StreamingDataWidthConverter_Batch_1_out_V_TREADY;
  wire StreamingDataWidthConverter_Batch_1_out_V_TVALID;
  wire [63:0]StreamingDataWidthConverter_Batch_2_out_V_TDATA;
  wire StreamingDataWidthConverter_Batch_2_out_V_TREADY;
  wire StreamingDataWidthConverter_Batch_2_out_V_TVALID;
  wire [1151:0]StreamingDataWidthConverter_Batch_3_out_V_TDATA;
  wire StreamingDataWidthConverter_Batch_3_out_V_TREADY;
  wire StreamingDataWidthConverter_Batch_3_out_V_TVALID;
  wire [7:0]StreamingDataWidthConverter_Batch_4_out_V_TDATA;
  wire StreamingDataWidthConverter_Batch_4_out_V_TREADY;
  wire StreamingDataWidthConverter_Batch_4_out_V_TVALID;
  wire [31:0]StreamingDataWidthConverter_Batch_5_out_V_TDATA;
  wire StreamingDataWidthConverter_Batch_5_out_V_TREADY;
  wire StreamingDataWidthConverter_Batch_5_out_V_TVALID;
  wire [2303:0]StreamingDataWidthConverter_Batch_6_out_V_TDATA;
  wire StreamingDataWidthConverter_Batch_6_out_V_TREADY;
  wire StreamingDataWidthConverter_Batch_6_out_V_TVALID;
  wire [15:0]StreamingDataWidthConverter_Batch_7_out_V_TDATA;
  wire StreamingDataWidthConverter_Batch_7_out_V_TREADY;
  wire StreamingDataWidthConverter_Batch_7_out_V_TVALID;
  wire [31:0]StreamingDataWidthConverter_Batch_8_out_V_TDATA;
  wire StreamingDataWidthConverter_Batch_8_out_V_TREADY;
  wire StreamingDataWidthConverter_Batch_8_out_V_TVALID;
  wire [7:0]StreamingFIFO_0_out_V_TDATA;
  wire StreamingFIFO_0_out_V_TREADY;
  wire StreamingFIFO_0_out_V_TVALID;
  wire [15:0]StreamingFIFO_12_out_V_TDATA;
  wire StreamingFIFO_12_out_V_TREADY;
  wire StreamingFIFO_12_out_V_TVALID;
  wire [7:0]StreamingFIFO_15_out_V_TDATA;
  wire StreamingFIFO_15_out_V_TREADY;
  wire StreamingFIFO_15_out_V_TVALID;
  wire [31:0]StreamingFIFO_17_out_V_TDATA;
  wire StreamingFIFO_17_out_V_TREADY;
  wire StreamingFIFO_17_out_V_TVALID;
  wire [7:0]StreamingFIFO_1_out_V_TDATA;
  wire StreamingFIFO_1_out_V_TREADY;
  wire StreamingFIFO_1_out_V_TVALID;
  wire [15:0]StreamingFIFO_22_out_V_TDATA;
  wire StreamingFIFO_22_out_V_TREADY;
  wire StreamingFIFO_22_out_V_TVALID;
  wire [15:0]StreamingFIFO_23_out_V_TDATA;
  wire StreamingFIFO_23_out_V_TREADY;
  wire StreamingFIFO_23_out_V_TVALID;
  wire [31:0]StreamingFIFO_26_out_V_TDATA;
  wire StreamingFIFO_26_out_V_TREADY;
  wire StreamingFIFO_26_out_V_TVALID;
  wire [71:0]StreamingFIFO_3_out_V_TDATA;
  wire StreamingFIFO_3_out_V_TREADY;
  wire StreamingFIFO_3_out_V_TVALID;
  wire [15:0]StreamingFIFO_6_out_V_TDATA;
  wire StreamingFIFO_6_out_V_TREADY;
  wire StreamingFIFO_6_out_V_TVALID;
  wire [63:0]StreamingFIFO_8_out_V_TDATA;
  wire StreamingFIFO_8_out_V_TREADY;
  wire StreamingFIFO_8_out_V_TVALID;
  wire [7:0]Thresholding_Batch_0_out_V_TDATA;
  wire Thresholding_Batch_0_out_V_TREADY;
  wire Thresholding_Batch_0_out_V_TVALID;
  wire [15:0]Thresholding_Batch_1_out_V_TDATA;
  wire Thresholding_Batch_1_out_V_TREADY;
  wire Thresholding_Batch_1_out_V_TVALID;
  wire [7:0]Thresholding_Batch_2_out_V_TDATA;
  wire Thresholding_Batch_2_out_V_TREADY;
  wire Thresholding_Batch_2_out_V_TVALID;
  wire ap_clk_0_1;
  wire ap_rst_n_0_1;
  wire [7:0]in0_V_0_1_TDATA;
  wire in0_V_0_1_TREADY;
  wire in0_V_0_1_TVALID;

  assign MatrixVectorActivation_3_out_V_TREADY = m_axis_0_tready;
  assign ap_clk_0_1 = ap_clk;
  assign ap_rst_n_0_1 = ap_rst_n;
  assign in0_V_0_1_TDATA = s_axis_0_tdata[7:0];
  assign in0_V_0_1_TVALID = s_axis_0_tvalid;
  assign m_axis_0_tdata[23:0] = MatrixVectorActivation_3_out_V_TDATA;
  assign m_axis_0_tvalid = MatrixVectorActivation_3_out_V_TVALID;
  assign s_axis_0_tready = in0_V_0_1_TREADY;
  finn_design_ConvolutionInputGenerator_0_0 ConvolutionInputGenerator_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_1_out_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_1_out_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_1_out_V_TVALID),
        .out_V_TDATA(ConvolutionInputGenerator_0_out_V_TDATA),
        .out_V_TREADY(ConvolutionInputGenerator_0_out_V_TREADY),
        .out_V_TVALID(ConvolutionInputGenerator_0_out_V_TVALID));
  finn_design_ConvolutionInputGenerator_1_0 ConvolutionInputGenerator_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_8_out_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_8_out_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_8_out_V_TVALID),
        .out_V_TDATA(ConvolutionInputGenerator_1_out_V_TDATA),
        .out_V_TREADY(ConvolutionInputGenerator_1_out_V_TREADY),
        .out_V_TVALID(ConvolutionInputGenerator_1_out_V_TVALID));
  finn_design_ConvolutionInputGenerator_2_0 ConvolutionInputGenerator_2
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_12_out_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_12_out_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_12_out_V_TVALID),
        .out_V_TDATA(ConvolutionInputGenerator_2_out_V_TDATA),
        .out_V_TREADY(ConvolutionInputGenerator_2_out_V_TREADY),
        .out_V_TVALID(ConvolutionInputGenerator_2_out_V_TVALID));
  finn_design_ConvolutionInputGenerator_3_0 ConvolutionInputGenerator_3
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_17_out_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_17_out_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_17_out_V_TVALID),
        .out_V_TDATA(ConvolutionInputGenerator_3_out_V_TDATA),
        .out_V_TREADY(ConvolutionInputGenerator_3_out_V_TREADY),
        .out_V_TVALID(ConvolutionInputGenerator_3_out_V_TVALID));
  finn_design_ConvolutionInputGenerator_4_0 ConvolutionInputGenerator_4
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_23_out_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_23_out_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_23_out_V_TVALID),
        .out_V_TDATA(ConvolutionInputGenerator_4_out_V_TDATA),
        .out_V_TREADY(ConvolutionInputGenerator_4_out_V_TREADY),
        .out_V_TVALID(ConvolutionInputGenerator_4_out_V_TVALID));
  finn_design_FMPadding_Batch_0_0 FMPadding_Batch_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_0_out_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_0_out_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_0_out_V_TVALID),
        .out_V_TDATA(FMPadding_Batch_0_out_V_TDATA),
        .out_V_TREADY(FMPadding_Batch_0_out_V_TREADY),
        .out_V_TVALID(FMPadding_Batch_0_out_V_TVALID));
  finn_design_FMPadding_Batch_1_0 FMPadding_Batch_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_6_out_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_6_out_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_6_out_V_TVALID),
        .out_V_TDATA(FMPadding_Batch_1_out_V_TDATA),
        .out_V_TREADY(FMPadding_Batch_1_out_V_TREADY),
        .out_V_TVALID(FMPadding_Batch_1_out_V_TVALID));
  finn_design_FMPadding_Batch_2_0 FMPadding_Batch_2
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_15_out_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_15_out_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_15_out_V_TVALID),
        .out_V_TDATA(FMPadding_Batch_2_out_V_TDATA),
        .out_V_TREADY(FMPadding_Batch_2_out_V_TREADY),
        .out_V_TVALID(FMPadding_Batch_2_out_V_TVALID));
  finn_design_FMPadding_Batch_3_0 FMPadding_Batch_3
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_22_out_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_22_out_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_22_out_V_TVALID),
        .out_V_TDATA(FMPadding_Batch_3_out_V_TDATA),
        .out_V_TREADY(FMPadding_Batch_3_out_V_TREADY),
        .out_V_TVALID(FMPadding_Batch_3_out_V_TVALID));
  MatrixVectorActivation_0_imp_MT0NP MatrixVectorActivation_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_tdata(StreamingFIFO_3_out_V_TDATA),
        .in0_V_tready(StreamingFIFO_3_out_V_TREADY),
        .in0_V_tvalid(StreamingFIFO_3_out_V_TVALID),
        .out_V_tdata(MatrixVectorActivation_0_out_V_TDATA),
        .out_V_tready(MatrixVectorActivation_0_out_V_TREADY),
        .out_V_tvalid(MatrixVectorActivation_0_out_V_TVALID));
  MatrixVectorActivation_1_imp_16PJ27U MatrixVectorActivation_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_tdata(StreamingDataWidthConverter_Batch_3_out_V_TDATA),
        .in0_V_tready(StreamingDataWidthConverter_Batch_3_out_V_TREADY),
        .in0_V_tvalid(StreamingDataWidthConverter_Batch_3_out_V_TVALID),
        .out_V_tdata(MatrixVectorActivation_1_out_V_TDATA),
        .out_V_tready(MatrixVectorActivation_1_out_V_TREADY),
        .out_V_tvalid(MatrixVectorActivation_1_out_V_TVALID));
  MatrixVectorActivation_2_imp_1U5U4NE MatrixVectorActivation_2
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_tdata(StreamingDataWidthConverter_Batch_6_out_V_TDATA),
        .in0_V_tready(StreamingDataWidthConverter_Batch_6_out_V_TREADY),
        .in0_V_tvalid(StreamingDataWidthConverter_Batch_6_out_V_TVALID),
        .out_V_tdata(MatrixVectorActivation_2_out_V_TDATA),
        .out_V_tready(MatrixVectorActivation_2_out_V_TREADY),
        .out_V_tvalid(MatrixVectorActivation_2_out_V_TVALID));
  MatrixVectorActivation_3_imp_WP21D1 MatrixVectorActivation_3
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_tdata(StreamingFIFO_26_out_V_TDATA),
        .in0_V_tready(StreamingFIFO_26_out_V_TREADY),
        .in0_V_tvalid(StreamingFIFO_26_out_V_TVALID),
        .out_V_tdata(MatrixVectorActivation_3_out_V_TDATA),
        .out_V_tready(MatrixVectorActivation_3_out_V_TREADY),
        .out_V_tvalid(MatrixVectorActivation_3_out_V_TVALID));
  finn_design_Pool_Batch_0_0 Pool_Batch_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(ConvolutionInputGenerator_2_out_V_TDATA),
        .in0_V_TREADY(ConvolutionInputGenerator_2_out_V_TREADY),
        .in0_V_TVALID(ConvolutionInputGenerator_2_out_V_TVALID),
        .out_V_TDATA(Pool_Batch_0_out_V_TDATA),
        .out_V_TREADY(Pool_Batch_0_out_V_TREADY),
        .out_V_TVALID(Pool_Batch_0_out_V_TVALID));
  finn_design_Pool_Batch_1_0 Pool_Batch_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(ConvolutionInputGenerator_4_out_V_TDATA),
        .in0_V_TREADY(ConvolutionInputGenerator_4_out_V_TREADY),
        .in0_V_TVALID(ConvolutionInputGenerator_4_out_V_TVALID),
        .out_V_TDATA(Pool_Batch_1_out_V_TDATA),
        .out_V_TREADY(Pool_Batch_1_out_V_TREADY),
        .out_V_TVALID(Pool_Batch_1_out_V_TVALID));
  finn_design_StreamingDataWidthConverter_Batch_0_0 StreamingDataWidthConverter_Batch_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(ConvolutionInputGenerator_0_out_V_TDATA),
        .in0_V_TREADY(ConvolutionInputGenerator_0_out_V_TREADY),
        .in0_V_TVALID(ConvolutionInputGenerator_0_out_V_TVALID),
        .out_V_TDATA(StreamingDataWidthConverter_Batch_0_out_V_TDATA),
        .out_V_TREADY(StreamingDataWidthConverter_Batch_0_out_V_TREADY),
        .out_V_TVALID(StreamingDataWidthConverter_Batch_0_out_V_TVALID));
  finn_design_StreamingDataWidthConverter_Batch_1_0 StreamingDataWidthConverter_Batch_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(Thresholding_Batch_0_out_V_TDATA),
        .in0_V_TREADY(Thresholding_Batch_0_out_V_TREADY),
        .in0_V_TVALID(Thresholding_Batch_0_out_V_TVALID),
        .out_V_TDATA(StreamingDataWidthConverter_Batch_1_out_V_TDATA),
        .out_V_TREADY(StreamingDataWidthConverter_Batch_1_out_V_TREADY),
        .out_V_TVALID(StreamingDataWidthConverter_Batch_1_out_V_TVALID));
  finn_design_StreamingDataWidthConverter_Batch_2_0 StreamingDataWidthConverter_Batch_2
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(FMPadding_Batch_1_out_V_TDATA),
        .in0_V_TREADY(FMPadding_Batch_1_out_V_TREADY),
        .in0_V_TVALID(FMPadding_Batch_1_out_V_TVALID),
        .out_V_TDATA(StreamingDataWidthConverter_Batch_2_out_V_TDATA),
        .out_V_TREADY(StreamingDataWidthConverter_Batch_2_out_V_TREADY),
        .out_V_TVALID(StreamingDataWidthConverter_Batch_2_out_V_TVALID));
  finn_design_StreamingDataWidthConverter_Batch_3_0 StreamingDataWidthConverter_Batch_3
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(ConvolutionInputGenerator_1_out_V_TDATA),
        .in0_V_TREADY(ConvolutionInputGenerator_1_out_V_TREADY),
        .in0_V_TVALID(ConvolutionInputGenerator_1_out_V_TVALID),
        .out_V_TDATA(StreamingDataWidthConverter_Batch_3_out_V_TDATA),
        .out_V_TREADY(StreamingDataWidthConverter_Batch_3_out_V_TREADY),
        .out_V_TVALID(StreamingDataWidthConverter_Batch_3_out_V_TVALID));
  finn_design_StreamingDataWidthConverter_Batch_4_0 StreamingDataWidthConverter_Batch_4
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(Pool_Batch_0_out_V_TDATA),
        .in0_V_TREADY(Pool_Batch_0_out_V_TREADY),
        .in0_V_TVALID(Pool_Batch_0_out_V_TVALID),
        .out_V_TDATA(StreamingDataWidthConverter_Batch_4_out_V_TDATA),
        .out_V_TREADY(StreamingDataWidthConverter_Batch_4_out_V_TREADY),
        .out_V_TVALID(StreamingDataWidthConverter_Batch_4_out_V_TVALID));
  finn_design_StreamingDataWidthConverter_Batch_5_0 StreamingDataWidthConverter_Batch_5
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(FMPadding_Batch_2_out_V_TDATA),
        .in0_V_TREADY(FMPadding_Batch_2_out_V_TREADY),
        .in0_V_TVALID(FMPadding_Batch_2_out_V_TVALID),
        .out_V_TDATA(StreamingDataWidthConverter_Batch_5_out_V_TDATA),
        .out_V_TREADY(StreamingDataWidthConverter_Batch_5_out_V_TREADY),
        .out_V_TVALID(StreamingDataWidthConverter_Batch_5_out_V_TVALID));
  finn_design_StreamingDataWidthConverter_Batch_6_0 StreamingDataWidthConverter_Batch_6
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(ConvolutionInputGenerator_3_out_V_TDATA),
        .in0_V_TREADY(ConvolutionInputGenerator_3_out_V_TREADY),
        .in0_V_TVALID(ConvolutionInputGenerator_3_out_V_TVALID),
        .out_V_TDATA(StreamingDataWidthConverter_Batch_6_out_V_TDATA),
        .out_V_TREADY(StreamingDataWidthConverter_Batch_6_out_V_TREADY),
        .out_V_TVALID(StreamingDataWidthConverter_Batch_6_out_V_TVALID));
  finn_design_StreamingDataWidthConverter_Batch_7_0 StreamingDataWidthConverter_Batch_7
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(Thresholding_Batch_2_out_V_TDATA),
        .in0_V_TREADY(Thresholding_Batch_2_out_V_TREADY),
        .in0_V_TVALID(Thresholding_Batch_2_out_V_TVALID),
        .out_V_TDATA(StreamingDataWidthConverter_Batch_7_out_V_TDATA),
        .out_V_TREADY(StreamingDataWidthConverter_Batch_7_out_V_TREADY),
        .out_V_TVALID(StreamingDataWidthConverter_Batch_7_out_V_TVALID));
  finn_design_StreamingDataWidthConverter_Batch_8_0 StreamingDataWidthConverter_Batch_8
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(Pool_Batch_1_out_V_TDATA),
        .in0_V_TREADY(Pool_Batch_1_out_V_TREADY),
        .in0_V_TVALID(Pool_Batch_1_out_V_TVALID),
        .out_V_TDATA(StreamingDataWidthConverter_Batch_8_out_V_TDATA),
        .out_V_TREADY(StreamingDataWidthConverter_Batch_8_out_V_TREADY),
        .out_V_TVALID(StreamingDataWidthConverter_Batch_8_out_V_TVALID));
  finn_design_StreamingFIFO_0_0 StreamingFIFO_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(in0_V_0_1_TDATA),
        .in0_V_TREADY(in0_V_0_1_TREADY),
        .in0_V_TVALID(in0_V_0_1_TVALID),
        .out_V_TDATA(StreamingFIFO_0_out_V_TDATA),
        .out_V_TREADY(StreamingFIFO_0_out_V_TREADY),
        .out_V_TVALID(StreamingFIFO_0_out_V_TVALID));
  finn_design_StreamingFIFO_1_0 StreamingFIFO_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(FMPadding_Batch_0_out_V_TDATA),
        .in0_V_TREADY(FMPadding_Batch_0_out_V_TREADY),
        .in0_V_TVALID(FMPadding_Batch_0_out_V_TVALID),
        .out_V_TDATA(StreamingFIFO_1_out_V_TDATA),
        .out_V_TREADY(StreamingFIFO_1_out_V_TREADY),
        .out_V_TVALID(StreamingFIFO_1_out_V_TVALID));
  StreamingFIFO_12_imp_Z91EFF StreamingFIFO_12
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_tdata(Thresholding_Batch_1_out_V_TDATA),
        .in0_V_tready(Thresholding_Batch_1_out_V_TREADY),
        .in0_V_tvalid(Thresholding_Batch_1_out_V_TVALID),
        .out_V_tdata(StreamingFIFO_12_out_V_TDATA),
        .out_V_tready(StreamingFIFO_12_out_V_TREADY),
        .out_V_tvalid(StreamingFIFO_12_out_V_TVALID));
  finn_design_StreamingFIFO_15_0 StreamingFIFO_15
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingDataWidthConverter_Batch_4_out_V_TDATA),
        .in0_V_TREADY(StreamingDataWidthConverter_Batch_4_out_V_TREADY),
        .in0_V_TVALID(StreamingDataWidthConverter_Batch_4_out_V_TVALID),
        .out_V_TDATA(StreamingFIFO_15_out_V_TDATA),
        .out_V_TREADY(StreamingFIFO_15_out_V_TREADY),
        .out_V_TVALID(StreamingFIFO_15_out_V_TVALID));
  finn_design_StreamingFIFO_17_0 StreamingFIFO_17
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingDataWidthConverter_Batch_5_out_V_TDATA),
        .in0_V_TREADY(StreamingDataWidthConverter_Batch_5_out_V_TREADY),
        .in0_V_TVALID(StreamingDataWidthConverter_Batch_5_out_V_TVALID),
        .out_V_TDATA(StreamingFIFO_17_out_V_TDATA),
        .out_V_TREADY(StreamingFIFO_17_out_V_TREADY),
        .out_V_TVALID(StreamingFIFO_17_out_V_TVALID));
  finn_design_StreamingFIFO_22_0 StreamingFIFO_22
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingDataWidthConverter_Batch_7_out_V_TDATA),
        .in0_V_TREADY(StreamingDataWidthConverter_Batch_7_out_V_TREADY),
        .in0_V_TVALID(StreamingDataWidthConverter_Batch_7_out_V_TVALID),
        .out_V_TDATA(StreamingFIFO_22_out_V_TDATA),
        .out_V_TREADY(StreamingFIFO_22_out_V_TREADY),
        .out_V_TVALID(StreamingFIFO_22_out_V_TVALID));
  StreamingFIFO_23_imp_OSV7QI StreamingFIFO_23
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_tdata(FMPadding_Batch_3_out_V_TDATA),
        .in0_V_tready(FMPadding_Batch_3_out_V_TREADY),
        .in0_V_tvalid(FMPadding_Batch_3_out_V_TVALID),
        .out_V_tdata(StreamingFIFO_23_out_V_TDATA),
        .out_V_tready(StreamingFIFO_23_out_V_TREADY),
        .out_V_tvalid(StreamingFIFO_23_out_V_TVALID));
  finn_design_StreamingFIFO_26_0 StreamingFIFO_26
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingDataWidthConverter_Batch_8_out_V_TDATA),
        .in0_V_TREADY(StreamingDataWidthConverter_Batch_8_out_V_TREADY),
        .in0_V_TVALID(StreamingDataWidthConverter_Batch_8_out_V_TVALID),
        .out_V_TDATA(StreamingFIFO_26_out_V_TDATA),
        .out_V_TREADY(StreamingFIFO_26_out_V_TREADY),
        .out_V_TVALID(StreamingFIFO_26_out_V_TVALID));
  finn_design_StreamingFIFO_3_0 StreamingFIFO_3
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingDataWidthConverter_Batch_0_out_V_TDATA),
        .in0_V_TREADY(StreamingDataWidthConverter_Batch_0_out_V_TREADY),
        .in0_V_TVALID(StreamingDataWidthConverter_Batch_0_out_V_TVALID),
        .out_V_TDATA(StreamingFIFO_3_out_V_TDATA),
        .out_V_TREADY(StreamingFIFO_3_out_V_TREADY),
        .out_V_TVALID(StreamingFIFO_3_out_V_TVALID));
  finn_design_StreamingFIFO_6_0 StreamingFIFO_6
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingDataWidthConverter_Batch_1_out_V_TDATA),
        .in0_V_TREADY(StreamingDataWidthConverter_Batch_1_out_V_TREADY),
        .in0_V_TVALID(StreamingDataWidthConverter_Batch_1_out_V_TVALID),
        .out_V_TDATA(StreamingFIFO_6_out_V_TDATA),
        .out_V_TREADY(StreamingFIFO_6_out_V_TREADY),
        .out_V_TVALID(StreamingFIFO_6_out_V_TVALID));
  finn_design_StreamingFIFO_8_0 StreamingFIFO_8
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingDataWidthConverter_Batch_2_out_V_TDATA),
        .in0_V_TREADY(StreamingDataWidthConverter_Batch_2_out_V_TREADY),
        .in0_V_TVALID(StreamingDataWidthConverter_Batch_2_out_V_TVALID),
        .out_V_TDATA(StreamingFIFO_8_out_V_TDATA),
        .out_V_TREADY(StreamingFIFO_8_out_V_TREADY),
        .out_V_TVALID(StreamingFIFO_8_out_V_TVALID));
  finn_design_Thresholding_Batch_0_0 Thresholding_Batch_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(MatrixVectorActivation_0_out_V_TDATA),
        .in0_V_TREADY(MatrixVectorActivation_0_out_V_TREADY),
        .in0_V_TVALID(MatrixVectorActivation_0_out_V_TVALID),
        .out_V_TDATA(Thresholding_Batch_0_out_V_TDATA),
        .out_V_TREADY(Thresholding_Batch_0_out_V_TREADY),
        .out_V_TVALID(Thresholding_Batch_0_out_V_TVALID));
  finn_design_Thresholding_Batch_1_0 Thresholding_Batch_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(MatrixVectorActivation_1_out_V_TDATA),
        .in0_V_TREADY(MatrixVectorActivation_1_out_V_TREADY),
        .in0_V_TVALID(MatrixVectorActivation_1_out_V_TVALID),
        .out_V_TDATA(Thresholding_Batch_1_out_V_TDATA),
        .out_V_TREADY(Thresholding_Batch_1_out_V_TREADY),
        .out_V_TVALID(Thresholding_Batch_1_out_V_TVALID));
  finn_design_Thresholding_Batch_2_0 Thresholding_Batch_2
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(MatrixVectorActivation_2_out_V_TDATA),
        .in0_V_TREADY(MatrixVectorActivation_2_out_V_TREADY),
        .in0_V_TVALID(MatrixVectorActivation_2_out_V_TVALID),
        .out_V_TDATA(Thresholding_Batch_2_out_V_TDATA),
        .out_V_TREADY(Thresholding_Batch_2_out_V_TREADY),
        .out_V_TVALID(Thresholding_Batch_2_out_V_TVALID));
endmodule
