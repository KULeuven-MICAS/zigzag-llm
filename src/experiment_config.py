import os

from zigzag.utils import pickle_deepcopy

from src.config import LLAMA_2_7B, OPT_125M, W32A32, W8I8O32
from src.util import Stage


class ExperimentConfig:
    def __init__(
        self,
        model=OPT_125M,
        stage=Stage.PREFILL,
        batch_size=1,
        prefill_size=256,
        decode_size=256,
        quant=W32A32,
        accelerator="generic_array_32b",
        mapping_path="inputs/mapping/weight_unrolled_256.yaml",
        out_path="outputs/main",
    ):
        self.model = model
        self.model.batch_size = batch_size
        self.model.prefill_size = prefill_size
        self.model.decode_size = decode_size
        self.stage = stage
        self.quant = quant
        self.accelerator = accelerator
        self.mapping_path = mapping_path
        self.out_path = out_path


## STANDARD AND SPECULATIVE DECODING EXPERIMENT PARAMETERS
# ACCELERATOR = "balanced_df_16_8_8"
# MAPPING = "inputs/mapping/balanced_df_16_8_8.yaml"
ACCELERATOR = "balanced_df_1_64_16"
MAPPING = "inputs/mapping/balanced_df_1_64_16.yaml"

## STANDARD DECODING EXPERIMENT CONFIGURATIONS
def get_standard_decoding_config_prefill(context_len, out_prefix):
    return ExperimentConfig(
        model=pickle_deepcopy(LLAMA_2_7B),
        stage=Stage.PREFILL,
        batch_size=1,
        prefill_size=context_len,
        decode_size=1,
        quant=W8I8O32,
        accelerator=ACCELERATOR,
        mapping_path=MAPPING,
        out_path=os.path.join(out_prefix, "standard/prefill/"),
    )


def get_standard_decoding_config_decode(context_len, decode_len, out_prefix):
    return ExperimentConfig(
        model=pickle_deepcopy(LLAMA_2_7B),
        stage=Stage.DECODE,
        batch_size=1,
        prefill_size=context_len,
        decode_size=decode_len,
        quant=W8I8O32,
        accelerator=ACCELERATOR,
        mapping_path=MAPPING,
        out_path=os.path.join(out_prefix, "standard/decode/"),
    )

