import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from custom_steps import custom_step_convert_to_hls
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn,
    step_tidy_up,
    step_streamline,
    step_create_dataflow_partition,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_generate_estimate_reports,
)

gen_estimate_steps = [
    step_qonnx_to_finn,
    step_tidy_up,
    step_streamline,
    custom_step_convert_to_hls,
    step_create_dataflow_partition,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_generate_estimate_reports,
]

model_file = "model.onnx"

my_output_dir = "output_estimates_only"

cfg = build_cfg.DataflowBuildConfig(
    steps=gen_estimate_steps,
    output_dir=my_output_dir,
    target_fps=100000,
    mvau_wwidth_max=10000,
    synth_clk_period_ns=10.0,
    board="Pynq-Z1",
    standalone_thresholds=True,
    shell_flow_type="vivado_zynq",
    verify_save_rtlsim_waveforms=True,
    enable_build_pdb_debug=True,
    verify_steps=[
        build_cfg.VerificationStepType.TIDY_UP_PYTHON,
        build_cfg.VerificationStepType.STREAMLINED_PYTHON,
        build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
    ],
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    ]
)
print("Running FINN flow: resource and performance estimate reports")
build.build_dataflow_cfg(model_file, cfg)

