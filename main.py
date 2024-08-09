from datetime import datetime
import os
import pickle
import time
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from src.export_onnx import export_transformer_to_onnx
from src.config import GPT3_175B, LLAMA_2_7B, OPT_125M, W16A32, W32A32, W4A16, W4A8, W8A8
from src.util import (
    get_cmes_full_model,
    get_cmes_to_plot,
)
from src.plots import (
    plot_energy_clean,
    plot_energy_minimal,
    plot_latency_clean,
)

model = OPT_125M
quant = W32A32
do_prefill = True
output_dir = "outputs/main"  # f"outputs/{datetime.now()}"
# workload_path = "inputs/workload/matmul.yaml"
accelerator_path = "inputs/hardware/generic_array_32b.yaml"
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"

workload_path = f"outputs/onnx/{model.parameterized_name}_{quant.name}_{'prefill' if do_prefill else 'decode'}.onnx"
pickle_filename = "outputs/TPU-cmes.pickle"


os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(workload_path):
    export_transformer_to_onnx(model.to_simulatable_config(), quant, path=workload_path, prefill=do_prefill)

start_time = time.time()
energy, latency, cmes = api.get_hardware_performance_zigzag(
    workload=workload_path,
    accelerator=accelerator_path,
    mapping=mapping_path,
    opt="EDP",
    dump_folder=output_dir,
    pickle_filename=pickle_filename,
    nb_spatial_mappings_generated=1,
)
end_time = time.time()

with open(pickle_filename, "rb") as fp:
    cmes = pickle.load(fp)

cmes_to_plot = get_cmes_to_plot(cmes)

bar_plot_cost_model_evaluations_breakdown(cmes, save_path=f"{output_dir}/all_layers.png")
bar_plot_cost_model_evaluations_breakdown(cmes_to_plot, save_path=f"{output_dir}/interesting_layers_single.png")

# Compute generalized results for full LLM
complete_result_cmes = get_cmes_full_model(cmes_to_plot, model)
bar_plot_cost_model_evaluations_breakdown(complete_result_cmes, save_path=f"{output_dir}/interesting_layers_full.png")
plot_energy_minimal(complete_result_cmes, f"{output_dir}/grouped_energy.png")
plot_latency_clean(complete_result_cmes, f"{output_dir}/grouped_latency.png")

print("Layers currently shown in plot:")
for idx, cme in enumerate(cmes_to_plot):
    print(f"\t{idx}: {cme.layer.name}")

print(f"Execution took {end_time - start_time} seconds")
