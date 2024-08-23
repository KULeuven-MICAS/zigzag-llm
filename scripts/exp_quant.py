"""
Run different quantization schemes
"""

import itertools
import os
import sys

sys.path.append(os.getcwd())
from src.config import OPT_125M, W1A32, W4A16, W32A32
from src.plots import (
    plot_energy_and_latency,
)
from src.simulation import run_simulation
from src.util import (
    CME_T,
    Stage,
    get_cmes_full_model_from_pickle,
    get_experiment_id,
)

model = OPT_125M
model.batch_size = 1
model.prefill_size = 256
model.decode_size = 256
quants = [W4A16, W1A32, W32A32]
accelerator = "generic_array_32b"
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"
out_path = "outputs/exp_quant"


def run_experiment():
    for quant, stage in itertools.product(quants, Stage):
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

    for quant in quants:
        cmes_per_group: list[list[CME_T]] = []

        for stage in Stage:
            experiment_name = get_experiment_id(model, stage, quant, accelerator)
            pickle_filename = f"{out_path}/{experiment_name}/cmes.pickle"
            cmes_full_model = get_cmes_full_model_from_pickle(pickle_filename, model, stage)
            cmes_per_group.append(cmes_full_model)

        plot_energy_and_latency(
            cmes_per_group,
            supergroups=["Prefill", "Decode"],
            title=f"{model.name} ({quant.name})",
            filename=f"{out_path}/energy_and_latency_{quant.name}_{model.name}.png",
        )
