
#vivado -mode batch -source <your_Tcl_script>
open_project ./../dataflow_build_dir/output_estimates_only/stitched_ip/finn_vivado_stitch_proj.xpr
#launch_runs synth_1 -jobs 24 -scripts_only
synth_design
#report_utilization -name utilization_1 GUI only
#report_utilization -hierarchical  -file location_of_the_report_file
report_utilization -name utilization_1 -spreadsheet_file util_table.xlsx -spreadsheet_table "Slice Logic - Slice LUT*"
report_utilization -name utilization_2 -spreadsheet_file util_table99.xlsx -spreadsheet_table "Slice Logic - Slice LUTs"
report_timing > timing.txt
close_project
exit


