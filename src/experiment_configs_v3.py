import os

from zigzag.utils import pickle_deepcopy

# from src.config import LLAMA_2_7B, OPT_125M, W32A32, W8I8O32
from src.config import LLAMA_3_1_8B, LLAMA_3_1_8B_EAGLE3_DRAFT, W32A32, W8I8O32
from src.util import Stage
from src.experiment_config import ExperimentConfig

## STANDARD AND SPECULATIVE DECODING EXPERIMENT PARAMETERS
# ACCELERATOR = "balanced_df_16_8_8"
# MAPPING = "inputs/mapping/balanced_df_16_8_8.yaml"
# ACCELERATOR = "balanced_df_1_64_16"
# MAPPING = "inputs/mapping/balanced_df_1_64_16.yaml"
ACCELERATOR = "balanced_df_16_16_16_new"
MAPPING = "inputs/mapping/balanced_df_16_16_16.yaml"

def get_speculative_draft_config_prefill(context_len, out_prefix):
    return ExperimentConfig(
        model=pickle_deepcopy(LLAMA_3_1_8B_EAGLE3_DRAFT),
        stage=Stage.PREFILL,
        batch_size=1,
        prefill_size=context_len,
        decode_size=1,
        quant=W8I8O32,
        accelerator=ACCELERATOR,
        mapping_path=MAPPING,
        out_path=os.path.join(out_prefix, "draft_prefill/"),
    )

def get_speculative_draft_config_decode(context_len, decode_len, out_prefix):
    return ExperimentConfig(
        model=pickle_deepcopy(LLAMA_3_1_8B_EAGLE3_DRAFT),
        stage=Stage.DECODE,
        batch_size=1,
        prefill_size=context_len,
        decode_size=decode_len,
        quant=W8I8O32,
        accelerator=ACCELERATOR,
        mapping_path=MAPPING,
        out_path=os.path.join(out_prefix, "draft_decode/"),
    )

# [TODO] Chao: change stage to VERIFY
def get_speculative_target_config_verify(context_len, out_prefix):
    # D. Target model prefill [ctx_len + decode_len] --> [ctx_len + decode_len + 1] (to get logits)
    return ExperimentConfig(
        model=pickle_deepcopy(LLAMA_3_1_8B),
        stage=Stage.PREFILL,
        batch_size=1,
        prefill_size=context_len,
        decode_size=1,
        quant=W8I8O32,
        accelerator=ACCELERATOR,
        mapping_path=MAPPING,
        out_path=os.path.join(out_prefix, "verification/"),
    )