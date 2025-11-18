import os
import json
from src.util import Stage, get_cmes_full_model_from_pickle
from src.config import LLAMA_2_7B
TARGET_MODEL = LLAMA_2_7B

def sum_cme_attribute(cmes_list, attribute):
    """Sum a given attribute across a list of CMEs."""
    return sum(getattr(cme, attribute) for cme in cmes_list)

def parse_standard_results(standard_path):
    """Parse the prefill and decode results for the standard output."""
    prefill_path = os.path.join(standard_path, "prefill")
    decode_path = os.path.join(standard_path, "decode")

    # Parse prefill CMEs
    prefill_cmes = []
    for run_folder in os.listdir(prefill_path):
        run_path = os.path.join(prefill_path, run_folder)
        cmes_path = os.path.join(run_path, "cmes.pickle")
        if os.path.exists(cmes_path):
            cmes = get_cmes_full_model_from_pickle(cmes_path, model=LLAMA_2_7B, stage=Stage.PREFILL)
            prefill_cmes += cmes

    # Parse decode CMEs
    decode_cmes = []
    for run_folder in os.listdir(decode_path):
        run_path = os.path.join(decode_path, run_folder)
        cmes_path = os.path.join(run_path, "cmes.pickle")
        if os.path.exists(cmes_path):
            cmes = get_cmes_full_model_from_pickle(cmes_path, model=LLAMA_2_7B, stage=Stage.DECODE)
            decode_cmes += cmes

    # Sum energy and latency
    total_energy = sum_cme_attribute(prefill_cmes, "energy_total") + sum_cme_attribute(decode_cmes, "energy_total")
    total_latency = sum_cme_attribute(prefill_cmes, "latency_total2") + sum_cme_attribute(decode_cmes, "latency_total2")

    return total_energy, total_latency

def write_standard_results(standard_path, energy, latency):
    """Write energy and latency for the standard output to a JSON file."""
    output_data = {
        "energy": energy,
        "latency": latency
    }
    output_file = os.path.join(standard_path, "performance.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Standard results written to {output_file}")

def main():
    STANDARD_PATH = "outputs/df_balanced_2_quant/standard/context_256_decode_256/standard/"

    energy, latency = parse_standard_results(STANDARD_PATH)
    print(f"Standard Energy: {energy:.2e}")
    print(f"Standard Latency: {latency:.2e}")

    write_standard_results(STANDARD_PATH, energy, latency)

if __name__ == "__main__":
    main()
