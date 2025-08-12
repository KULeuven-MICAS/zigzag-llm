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
    get_speculative_draft_config,
    get_speculative_target_fallback_config,
    get_speculative_target_verification_config,
)
from src.simulation import run_simulation

CONTEXT_LEN = 8
DECODE_LEN = 8
OUT_PREFIX = f"outputs/context_{CONTEXT_LEN}_decode_{DECODE_LEN}/"
if DECODE_LEN == 8:
    K_RANGE = [0, 1, 2, 3, 4, 5, 6, 7]
elif DECODE_LEN == 256:
    K_RANGE = [0, 1, 63, 127, 127 + 64, 255]
else:
    raise ValueError("Unsupported DECODE_LEN value")


def run_experiment():
    # Run Draft
    draft_config = get_speculative_draft_config(
        context_len=CONTEXT_LEN,
        decode_len=DECODE_LEN,
        out_prefix=OUT_PREFIX,
    )
    run_simulation(
        model=draft_config.model,
        stage=draft_config.stage,
        quant=draft_config.quant,
        accelerator_name=draft_config.accelerator,
        mapping_path=draft_config.mapping_path,
        output_dir=draft_config.out_path,
    )
    # Run Verification
    verification_config = get_speculative_target_verification_config(
        context_len=CONTEXT_LEN, out_prefix=OUT_PREFIX
    )
    run_simulation(
        model=verification_config.model,
        stage=verification_config.stage,
        quant=verification_config.quant,
        accelerator_name=verification_config.accelerator,
        mapping_path=verification_config.mapping_path,
        output_dir=verification_config.out_path,
    )
    # Run Fallback
    for k in K_RANGE:
        prefill_size = CONTEXT_LEN + k
        decode_size = DECODE_LEN - k
        fallback_config = get_speculative_target_fallback_config(
            context_len=prefill_size, decode_len=decode_size, out_prefix=OUT_PREFIX
        )
        run_simulation(
            model=fallback_config.model,
            stage=fallback_config.stage,
            quant=fallback_config.quant,
            accelerator_name=fallback_config.accelerator,
            mapping_path=fallback_config.mapping_path,
            output_dir=fallback_config.out_path,
        )


if __name__ == "__main__":
    os.makedirs(OUT_PREFIX, exist_ok=True)
    run_experiment()
