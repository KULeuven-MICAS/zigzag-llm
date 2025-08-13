import os

from zigzag.utils import pickle_deepcopy

from src.config import LLAMA_2_7B, OPT_125M, W32A32
from src.util import Stage
from src.experiment_config import ExperimentConfig

## STANDARD AND SPECULATIVE DECODING EXPERIMENT PARAMETERS
ACCELERATOR = "generic_array_32b"
MAPPING = "inputs/mapping/weight_unrolled_256.yaml"

def get_speculative_draft_config_prefill(context_len, out_prefix):
    return ExperimentConfig(
        model=pickle_deepcopy(OPT_125M),
        stage=Stage.PREFILL,
        batch_size=1,
        prefill_size=context_len,
        decode_size=1,
        quant=W32A32,
        accelerator=ACCELERATOR,
        mapping_path=MAPPING,
        out_path=os.path.join(out_prefix, "draft/"),
    )

def get_speculative_draft_config_decode(context_len, decode_len, out_prefix):
    return ExperimentConfig(
        model=pickle_deepcopy(OPT_125M),
        stage=Stage.DECODE,
        batch_size=1,
        prefill_size=context_len,
        decode_size=decode_len,
        quant=W32A32,
        accelerator=ACCELERATOR,
        mapping_path=MAPPING,
        out_path=os.path.join(out_prefix, "draft/"),
    )

def get_speculative_target_verification_config(context_len, out_prefix):
    # D. Target model prefill [ctx_len + decode_len] --> [ctx_len + decode_len + 1] (to get logits)
    return ExperimentConfig(
        model=pickle_deepcopy(LLAMA_2_7B),
        stage=Stage.PREFILL,
        batch_size=1,
        prefill_size=context_len,
        decode_size=1,
        quant=W32A32,
        accelerator=ACCELERATOR,
        mapping_path=MAPPING,
        out_path=os.path.join(out_prefix, "verification/"),
    )