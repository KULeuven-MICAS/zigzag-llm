"""
Evaluate a range of models with different accelerators, quantization schemes, batch sizes, ...
"""

import itertools
import os
import sys

sys.path.append(os.getcwd())
from src.simulation import run_simulation
from src.config import ALL_MODELS, BATCH_SIZE, W4A16, W8A8
from src.util import Stage


models = ALL_MODELS
quants = [W8A8]
accelerators = ["generic_array_32b", "generic_array_edge_32b"]
batch_sizes = [BATCH_SIZE]
mapping_path = "inputs/mapping/output_unrolled_256.yaml"
out_path = "outputs/experiments"

if __name__ == "__main__":
    for model, accelerator, quant, batch_size, stage in itertools.product(
        models, accelerators, quants, batch_sizes, Stage
    ):

        # Overwrite batch size for the experiment
        model.batch_size = batch_size

        run_simulation(
            model=model,
            stage=stage,
            quant=quant,
            accelerator_name=accelerator,
            mapping_path=mapping_path,
            output_dir=out_path,
        )
