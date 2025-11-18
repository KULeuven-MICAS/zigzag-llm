"""
Run different simulations to emulate the performance of speculative decoding
Target model: LLama2-7B
Draft model: OPT-125M
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
import random
import sys

sys.path.append(os.getcwd())
# import speculative experiment configs
from src.experiment_configs_v2 import (
    get_speculative_draft_config_prefill,
    get_speculative_draft_config_decode,
    get_speculative_target_verification_config,
)
from src.simulation import run_simulation

NB_SAMPLES = 16
CONTEXT_LEN = 256
DECODE_LEN = 256
DRAFT_DECODE_LEN = 5
OUT_PREFIX = f"outputs/df_balanced_2_quant/initial_context_{CONTEXT_LEN}_to_decode_{DECODE_LEN}_total_with_{DRAFT_DECODE_LEN}_draft_decode/"

def sample_accepted_tokens(max_tokens=5, p=0.8):
    for k in range(max_tokens):
        if random.random() >= p:
            return k
    return max_tokens  # all tokens matched

def run_experiment(out_prefix):
    # Run initial Draft prefill
    draft_prefill_config = get_speculative_draft_config_prefill(
        context_len=CONTEXT_LEN,
        out_prefix=out_prefix,
    )
    run_simulation(
        model=draft_prefill_config.model,
        stage=draft_prefill_config.stage,
        quant=draft_prefill_config.quant,
        accelerator_name=draft_prefill_config.accelerator,
        mapping_path=draft_prefill_config.mapping_path,
        output_dir=draft_prefill_config.out_path,
    )
    total_nb_accepted_tokens = 0
    context_len = CONTEXT_LEN
    i = 0
    while context_len < CONTEXT_LEN + DECODE_LEN:
        # Do draft speculation (decode)
        draft_decode_config = get_speculative_draft_config_decode(
            context_len=context_len, decode_len=DRAFT_DECODE_LEN, out_prefix=out_prefix
        )
        run_simulation(
            model=draft_decode_config.model,
            stage=draft_decode_config.stage,
            quant=draft_decode_config.quant,
            accelerator_name=draft_decode_config.accelerator,
            mapping_path=draft_decode_config.mapping_path,
            output_dir=draft_decode_config.out_path,
        )
        # Do target verification (prefill)
        target_verification_config = get_speculative_target_verification_config(
            context_len=context_len, out_prefix=out_prefix
        )
        run_simulation(
            model=target_verification_config.model,
            stage=target_verification_config.stage,
            quant=target_verification_config.quant,
            accelerator_name=target_verification_config.accelerator,
            mapping_path=target_verification_config.mapping_path,
            output_dir=target_verification_config.out_path,
        )
        # Choose a random number of accepted tokens from NB_TOKENS_VERIFIED
        nb_accepted_tokens = sample_accepted_tokens()
        total_nb_accepted_tokens += nb_accepted_tokens
        context_len += nb_accepted_tokens + 1  # +1 for next target model prediction (otherwise keep speculating same draft tokens)

        print(f"Iteration {i}: Accepted {nb_accepted_tokens} tokens, total accepted: {total_nb_accepted_tokens}, new context length: {context_len}")
        i += 1

if __name__ == "__main__":
    for sample in range(0, NB_SAMPLES):
        print(f"Launching sample {sample}")
        out_prefix = os.path.join(OUT_PREFIX, f"sample_{sample}")
        os.makedirs(out_prefix, exist_ok=True)
        run_experiment(out_prefix)
