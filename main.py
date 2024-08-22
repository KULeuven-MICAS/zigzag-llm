from src.config import OPT_125M, W32A32
from src.simulation import run_simulation
from src.util import Stage

model = OPT_125M
quant = W32A32
stage = Stage.PREFILL
accelerator = "generic_array_32b"
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"
out_path = "outputs/main"


if __name__ == "__main__":
    run_simulation(
        model=model,
        stage=stage,
        quant=quant,
        accelerator_name=accelerator,
        mapping_path=mapping_path,
        output_dir=out_path,
    )
