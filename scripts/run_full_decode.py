"""
Simulate the decoding phase by evaluating each token separately in the sequence from L/2 to L.
"""

import json
import os
import sys

import imageio
from zigzag import api

sys.path.append(os.getcwd())
from src.config import LLAMA_1_7B, W8A8
from src.export_onnx import export_transformer_to_onnx
from src.util import Stage, get_accelerator_path

model = LLAMA_1_7B
quant = W8A8
# workload_path = "inputs/workload/matmul.yaml"
accelerator = "generic_array_16b"
mapping_path = "inputs/mapping/weight_st_256.yaml"
pickle_filename = "outputs/TPU-cmes.pickle"
out_path = "outputs/full_decode"


def run_experiment():
    for decode_idx in range(model.seq_len // 2 + 1, model.seq_len):
        # Overwrite decode_idx
        model.decode_idx = decode_idx

        output_dir = f"outputs/full_decode/{model.name}_{quant.name}_decode={decode_idx}"
        workload_path = f"outputs/full_decode/onnx/{model.name}_{quant.name}_decode={decode_idx}.onnx"

        try:
            if not os.path.exists(workload_path):
                export_transformer_to_onnx(model.to_simulatable_config(), quant, path=workload_path, stage=Stage.DECODE)

            api.get_hardware_performance_zigzag(
                workload=workload_path,
                accelerator=get_accelerator_path(accelerator),
                mapping=mapping_path,
                opt="energy",
                dump_folder=output_dir,
                pickle_filename=pickle_filename,
                nb_spatial_mappings_generated=1,
            )
        except:
            continue


def make_gif():
    image_list = [
        f"outputs/full_decode/{model.name}_{quant.name}_decode={idx}/interesting_layers_full.png"
        for idx in range(1025, 2048)
    ]

    output_gif = "outputs/decoding.gif"

    images = [imageio.imread(file) for file in image_list]
    gif_duration = 10  # in seconds
    time_per_frame = gif_duration / len(image_list)
    # duration is the time between frames in seconds
    imageio.mimsave(output_gif, images, duration=time_per_frame)


if __name__ == "__main__":
    # run_experiment()

    result_list = [
        f"outputs/full_decode/{model.name}_{quant.name}_decode={idx}/overall_simple.json" for idx in range(1025, 2048)
    ]

    energy_list = [json.load(open(file))["energy"] for file in result_list]

    total_energy = sum(energy_list)
    total_energy_approx = energy_list[len(energy_list) // 2] * len(energy_list)

    print(f"Total energy {total_energy:.6e} approximated by {total_energy_approx:.6e}")
    print(f"Relative error: {(abs(total_energy - total_energy_approx) / total_energy)}")
