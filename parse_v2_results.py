import os
import json
from src.experiment_configs_v2 import (
    get_speculative_draft_config_prefill,
    get_speculative_draft_config_decode,
    get_speculative_target_verification_config,
)
from src.util import Stage, get_cmes_full_model_from_pickle
from src.plots import plot_energy_and_latency

from src.config import LLAMA_2_7B, OPT_125M
LLM_CONFIG_MAP = {'Llama2': LLAMA_2_7B, 'OPT-125M': OPT_125M}
TARGET_MODEL = LLAMA_2_7B
DRAFT_MODEL = OPT_125M

def sum_cme_attribute(cmes_list, attribute):
    """Sum a given attribute across a list of CMEs."""
    return sum(getattr(cme, attribute) for cme in cmes_list)

def parse_sample_results(sample_path):
    """Parse the draft and verification results for a single sample."""
    draft_path = os.path.join(sample_path, "draft")
    verification_path = os.path.join(sample_path, "verification")

    # Parse draft CMEs
    draft_cmes = []
    for run_folder in os.listdir(draft_path):
        run_path = os.path.join(draft_path, run_folder)
        cmes_path = os.path.join(run_path, "cmes.pickle")
        if 'prefill' in run_folder:
            stage = Stage.PREFILL
        else:
            stage = Stage.DECODE
        if os.path.exists(cmes_path):
            cmes = get_cmes_full_model_from_pickle(cmes_path, DRAFT_MODEL, stage)
            draft_cmes += cmes

    # Parse verification CMEs
    verification_cmes = []
    for run_folder in os.listdir(verification_path):
        run_path = os.path.join(verification_path, run_folder)
        cmes_path = os.path.join(run_path, "cmes.pickle")
        if 'prefill' in run_folder:
            stage = Stage.PREFILL
        else:
            stage = Stage.DECODE
        if os.path.exists(cmes_path):
            cmes = get_cmes_full_model_from_pickle(cmes_path, TARGET_MODEL, stage)
            verification_cmes += cmes

    # Sum energy and latency
    total_energy = sum_cme_attribute(draft_cmes, "energy_total") + sum_cme_attribute(verification_cmes, "energy_total")
    total_latency = sum_cme_attribute(draft_cmes, "latency_total2") + sum_cme_attribute(verification_cmes, "latency_total2")

    return total_energy, total_latency

def write_sample_results(sample_path, energy, latency):
    """Write energy and latency for a single sample to a JSON file."""
    sample_output_data = {
        "energy": energy,
        "latency": latency
    }
    sample_output_file = os.path.join(sample_path, "sample_performance.json")
    with open(sample_output_file, "w") as f:
        json.dump(sample_output_data, f, indent=4)
    print(f"Sample results written to {sample_output_file}")

def calculate_and_write_averages(out_prefix, total_energy, total_latency, num_samples):
    """Calculate and write average energy and latency across all samples to a JSON file."""
    avg_energy = total_energy / num_samples if num_samples > 0 else 0
    avg_latency = total_latency / num_samples if num_samples > 0 else 0

    avg_output_data = {
        "average_energy": avg_energy,
        "average_latency": avg_latency
    }
    avg_output_file = os.path.join(out_prefix, "average_performance.json")
    with open(avg_output_file, "w") as f:
        json.dump(avg_output_data, f, indent=4)
    print(f"Average results written to {avg_output_file}")

def process_samples(out_prefix):
    """Process all samples in the output directory."""
    total_energy_all_samples = 0
    total_latency_all_samples = 0
    num_samples = 0

    for sample_folder in sorted(os.listdir(out_prefix)):
        sample_path = os.path.join(out_prefix, sample_folder)
        if os.path.isdir(sample_path):
            energy, latency = parse_sample_results(sample_path)
            total_energy_all_samples += energy
            total_latency_all_samples += latency
            num_samples += 1
            print(f"Sample {sample_folder}: Energy = {energy:.2e}, Latency = {latency:.2e}")
            write_sample_results(sample_path, energy, latency)

    return total_energy_all_samples, total_latency_all_samples, num_samples

def main():
    CONTEXT_LEN = 256
    DECODE_LEN = 256
    DRAFT_DECODE_LEN = 5
    OUT_PREFIX = f"outputs/v2/initial_context_{CONTEXT_LEN}_to_decode_{DECODE_LEN}_total_with_{DRAFT_DECODE_LEN}_draft_decode/"

    total_energy, total_latency, num_samples = process_samples(OUT_PREFIX)
    print(f"Total Energy (All Samples): {total_energy:.2e}")
    print(f"Total Latency (All Samples): {total_latency:.2e}")

    calculate_and_write_averages(OUT_PREFIX, total_energy, total_latency, num_samples)

if __name__ == "__main__":
    main()
