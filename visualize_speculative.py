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
    CONTEXT_LEN,
    DECODE_LEN,
    get_speculative_target_fallback_config,
    # get_speculative_a_config,
    get_speculative_target_verification_config,
)
from src.experiment_visualize import visualize_experiment

if __name__ == "__main__":
    # # Plot A
    # speculative_a_config = get_speculative_a_config()
    # supergroups = ["Prefill"]
    # fig_title = f"Target Prefill ({speculative_a_config.quant.name})"
    # fig_path = f"{speculative_a_config.out_path}/plot.png"
    # visualize_experiment(
    #     speculative_a_config,
    #     supergroups,
    #     speculative_a_config.out_path,
    #     fig_title=fig_title,
    #     fig_path=fig_path,
    # )
    # Plot D
    speculative_d_config = get_speculative_target_verification_config()
    supergroups = ["Decode"]
    fig_title = f"Target Verification ({speculative_d_config.quant.name})"
    fig_path = f"{speculative_d_config.out_path}/plot.png"
    visualize_experiment(
        speculative_d_config,
        supergroups,
        fig_title=fig_title,
        out_path=speculative_d_config.out_path,
        fig_path=fig_path,
    )
    # Plot E
    speculative_e_config = get_speculative_target_fallback_config()
    supergroups = ["Decode"]
    for k in range(0, DECODE_LEN):
        prefill_size = CONTEXT_LEN + k
        decode_size = DECODE_LEN - k
        # Set the correct prefill, decode size and out_path of the experiment
        speculative_e_config.model.prefill_size = prefill_size
        speculative_e_config.model.decode_size = decode_size
        out_path = os.path.join(speculative_e_config.out_path, f"k_{k}")
        fig_title = f"Target Fallback (k={k}) ({speculative_e_config.quant.name})"
        fig_path = f"{out_path}/plot.png"
        visualize_experiment(
            speculative_e_config,
            supergroups,
            out_path=out_path,
            fig_title=fig_title,
            fig_path=fig_path,
        )
