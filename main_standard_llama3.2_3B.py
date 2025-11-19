"""
Run different simulations to emulate the performance of speculative decoding
Target model: LLama2-7B
Draft model: assumed negligible impact on performance
Context length: variable
Decode length: variable

This file currently simulates the following stages:
A. Target model prefill [ctx_len] --> [ctx_len + 1] (for target decode later)
B. [OMITTED] Draft model prefill [ctx_len] --> [ctx_len + 1] (prefill before draft generation)
C. [OMITTED] Draft model decode [ctx_len] --> [ctx_len + decode_len] (draft generation)
D. Target model decode [ctx_len + decode_len] --> [ctx_len + decode_len + 1] (to get logits)
E. Target decode [ctx_len + k] --> [ctx_len + decode_len - k] for k in range(decode_len)
"""

import os
import sys

sys.path.append(os.getcwd())
# import speculative experiment configs
from src.experiment_config import (
    get_llama3_decoding_config_decode,
    get_llama3_decoding_config_prefill,
)
from src.simulation import run_simulation

CONTEXT_LEN = 4096
DECODE_LEN = 1
OUT_PREFIX = f"outputs/llama3_unlimited_bw_even/"


def run_experiment():
    # Run prefill
    standard_prefill_config = get_llama3_decoding_config_prefill(
        context_len=CONTEXT_LEN, out_prefix=OUT_PREFIX
    )
    run_simulation(
        model=standard_prefill_config.model,
        stage=standard_prefill_config.stage,
        quant=standard_prefill_config.quant,
        accelerator_name=standard_prefill_config.accelerator,
        mapping_path=standard_prefill_config.mapping_path,
        output_dir=standard_prefill_config.out_path,
    )

    # Run decode
    standard_decode_config = get_llama3_decoding_config_decode(
        context_len=CONTEXT_LEN, out_prefix=OUT_PREFIX
    )
    run_simulation(
        model=standard_decode_config.model,
        stage=standard_decode_config.stage,
        quant=standard_decode_config.quant,
        accelerator_name=standard_decode_config.accelerator,
        mapping_path=standard_decode_config.mapping_path,
        output_dir=standard_decode_config.out_path,
    )

if __name__ == "__main__":
    run_experiment()
