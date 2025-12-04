"""
Run different simulations to emulate the performance of speculative decoding
Target model: Llama3.1-8B
Draft model: Llama3.1-8B-Eagle3-Draft
Context length: variable
Decode length: variable

This file currently simulates the following stages:
A. Target model prefill [ctx_len] --> [ctx_len + 1] (for target decode later)
B. [OMITTED] Draft model prefill [ctx_len] --> [ctx_len + 1] (prefill before draft generation)
C. [OMITTED] Draft model decode [ctx_len] --> [ctx_len + decode_len] (draft generation)
D. Target model decode [ctx_len + decode_len] --> [ctx_len + decode_len + 1] (to get logits)
E. Target decode [ctx_len + k] --> [ctx_len + decode_len - k] for k in range(decode_len)
"""

import argparse
import os
import random
import sys

sys.path.append(os.getcwd())
# import speculative experiment configs
from src.experiment_configs_v3 import (
    get_speculative_draft_config_prefill,
    get_speculative_draft_config_decode,
    get_speculative_target_config_verify,
)
from src.simulation import run_simulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Speculative decoding simulation with Eagle3 draft model"
    )
    parser.add_argument(
        "--nb_samples",
        type=int,
        default=16,
        help="Number of samples to run (default: 16)",
    )
    parser.add_argument(
        "--prefill_len",
        type=int,
        default=256,
        help="Initial prefill context length (default: 256)",
    )
    parser.add_argument(
        "--decode_len",
        type=int,
        default=256,
        help="Total decode length (default: 256)",
    )
    # [TODO] Chao: Here we assume it geneaates consecutive tokens each time, while a tree is actually generated.
    #              We need to two arguments to present this. One is nb_tokens_drafted, the other is nb_tokens_verified.
    parser.add_argument(
        "--draft_len_per_verify",
        type=int,
        default=5,
        help="Draft decode length per verification iteration (default: 5)",
    )
    parser.add_argument(
        "--proj_name",
        type=str,
        default="eagle3_sim_v1",
        help="Project name for output directory (default: eagle3_sim_v1)",
    )
    parser.add_argument(
        "--accept_prob",
        type=float,
        default=0.8,
        help="Token acceptance probability in sampling (default: 0.8)",
    )
    return parser.parse_args()


def sample_accepted_tokens(max_tokens, p):
    for k in range(max_tokens):
        if random.random() >= p:
            return k
    return max_tokens  # all tokens matched


def run_experiment(out_prefix, prefill_len, decode_len, draft_len_per_verify, accept_prob):
    # Run initial Draft prefill
    draft_prefill_config = get_speculative_draft_config_prefill(
        context_len=prefill_len,
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
    current_context_len = prefill_len
    i = 0
    
    while current_context_len < prefill_len + decode_len:
        # Do draft speculation (decode)
        draft_decode_config = get_speculative_draft_config_decode(
            context_len=current_context_len,
            decode_len=draft_len_per_verify,
            out_prefix=out_prefix,
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
        target_verification_config = get_speculative_target_config_verify(
            context_len=current_context_len + draft_len_per_verify,
            out_prefix=out_prefix,
        )
        # [TODO] Chao: change to get_speculative_target_config_verify (under development)
        # target_verification_config = get_speculative_target_config_verify(
        #     context_len=current_context_len, decode_len=draft_len_per_verify, out_prefix=out_prefix
        # )
        run_simulation(
            model=target_verification_config.model,
            stage=target_verification_config.stage,
            quant=target_verification_config.quant,
            accelerator_name=target_verification_config.accelerator,
            mapping_path=target_verification_config.mapping_path,
            output_dir=target_verification_config.out_path,
        )
        
        # Choose a random number of accepted tokens from NB_TOKENS_VERIFIED
        nb_accepted_tokens = sample_accepted_tokens(
            max_tokens=draft_len_per_verify, p=accept_prob
        )
        total_nb_accepted_tokens += nb_accepted_tokens
        current_context_len += nb_accepted_tokens + 1  # +1 for next target model prediction

        print(
            f"Iteration {i}: Accepted {nb_accepted_tokens} tokens, "
            f"total accepted: {total_nb_accepted_tokens}, "
            f"new context length: {current_context_len}"
        )
        i += 1


if __name__ == "__main__":
    args = parse_args()
    
    # Generate output prefix based on parameters
    out_prefix_base = (
        f"outputs/{args.proj_name}/initial_context_{args.prefill_len}_"
        f"to_decode_{args.decode_len}_total_with_{args.draft_len_per_verify}_draft_decode/"
    )
    
    for sample in range(args.nb_samples):
        print(f"Launching sample {sample}")
        out_prefix = os.path.join(out_prefix_base, f"sample_{sample}")
        os.makedirs(out_prefix, exist_ok=True)
        run_experiment(
            out_prefix=out_prefix,
            prefill_len=args.prefill_len,
            decode_len=args.decode_len,
            draft_len_per_verify=args.draft_len_per_verify,
            accept_prob=args.accept_prob,
        )