"""
Make plots to compare different architectures on the same models
"""

import itertools
import os
import sys

sys.path.append(os.getcwd())
from src.config import LLAMA_2_7B, W4A16
from src.plots import (
    plot_energy_and_latency_minimal,
)
from src.simulation import run_simulation
from src.util import (
    CME_T,
    Stage,
    get_cmes_full_model_from_pickle,
    get_experiment_id,
)

models = [LLAMA_2_7B]
quant = W4A16
accelerators = ["generic_array_32b", "generic_array_edge_32b"]
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"
out_path = "outputs/exp_compare_arch"


def run_experiment():
    for model, accelerator, stage in itertools.product(models, accelerators, Stage):
        run_simulation(
            model=model,
            stage=stage,
            quant=quant,
            accelerator_name=accelerator,
            mapping_path=mapping_path,
            output_dir=out_path,
        )


if __name__ == "__main__":
    run_experiment()

    for model in models:
        cmes_per_arch: list[list[CME_T]] = []

        for accelerator, stage in itertools.product(accelerators, Stage):
            experiment_name = get_experiment_id(model, stage, quant, accelerator)
            pickle_filename = f"{out_path}/{experiment_name}/cmes.pickle"
            cmes_full_model = get_cmes_full_model_from_pickle(pickle_filename, model, stage)
            cmes_per_arch.append(cmes_full_model)

        groups = ["Cloud\nprefill", "Cloud\ndecode", "Edge\nprefill", "Edge\ndecode"]

        plot_energy_and_latency_minimal(
            cmes_per_arch,
            groups=groups,
            title=f"{model.name} ({quant.name})",
            filename=f"{out_path}/compare_energy_and_latency_{model.name}.png",
        )
