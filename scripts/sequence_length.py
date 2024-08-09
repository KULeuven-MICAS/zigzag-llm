"""
Show what happens if the sequence length becomes very small / large.
"""

from copy import deepcopy
import itertools
import os
import sys

sys.path.append(os.getcwd())
from src.simulation import run_simulation
from src.config import ALL_MODELS, BATCH_SIZE, LLAMA_2_7B, PAPER_MODELS, W4A16, W8A8, LLMConfig
from src.util import (
    CME_T,
    Stage,
    get_cmes_full_model_from_pickle,
)
from src.plots import (
    plot_energy_and_latency,
)

models = [LLAMA_2_7B]
quant = W4A16
accelerators = ["generic_array_edge_16b"]
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"
out_path = "outputs/sequence_length"

scenarios = [
    (64, 64),
    (128, 128),
    (16_384, 64),
    (128, 16_384),
]

modified_models: list[LLMConfig] = []
for model, scenario in itertools.product(models, scenarios):
    # Create new config and override sequence length
    prefill_len, decode_len = scenario
    modified_model = deepcopy(model)
    modified_model.prefill_size = prefill_len
    modified_model.decode_size = decode_len
    modified_models.append(modified_model)


def run_experiment():
    for modified_model, accelerator, stage in itertools.product(modified_models, accelerators, Stage):

        identifier = (
            f"{modified_model.parameterized_name}_{quant.name}_"
            f"{f'prefill={modified_model.prefill_size}' if stage == Stage.PREFILL else f'decode={modified_model.decode_idx}'}"
        )
        experiment_name = f"{identifier}_{accelerator}"
        onnx_path = f"outputs/onnx/{identifier}.onnx"

        run_simulation(
            model=model,
            stage=stage,
            quant=quant,
            accelerator_name=accelerator,
            mapping_path=mapping_path,
            output_dir=out_path,
            experiment_id=experiment_name,
            onnx_path=onnx_path,
        )


if __name__ == "__main__":
    run_experiment()

    for modified_model in modified_models:

        cmes_per_arch: list[list[CME_T]] = []

        for accelerator, stage in itertools.product(accelerators, Stage):
            identifier = (
                f"{modified_model.parameterized_name}_{quant.name}_"
                f"{f'prefill={modified_model.prefill_size}' if stage == Stage.PREFILL else f'decode={modified_model.decode_idx}'}"
            )
            experiment_name = f"{identifier}_{accelerator}"
            pickle_filename = f"{out_path}/{experiment_name}/cmes.pickle"
            cmes_full_model = get_cmes_full_model_from_pickle(pickle_filename, modified_model, stage)
            cmes_per_arch.append(cmes_full_model)

        groups = ["Prefill", "Decode"]
        assert len(groups) == len(cmes_per_arch)

        plot_energy_and_latency(
            cmes_per_arch,
            supergroups=groups,
            title=modified_model.name,
            filename=f"{out_path}/{modified_model.name}_({modified_model.prefill_size}_{modified_model.decode_size}).png",
        )
