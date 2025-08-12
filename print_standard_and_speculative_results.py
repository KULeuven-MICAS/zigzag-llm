import json

from src.experiment_config import (
    get_speculative_draft_config,
    get_speculative_target_fallback_config,
    get_speculative_target_verification_config,
    get_standard_decoding_config_decode,
    get_standard_decoding_config_prefill,
)
from src.experiment_results import SimpleResults
from src.plots import plot_energy_and_latency
from src.speculative.probabilities import (
    exp_decay_acceptance_pdf,
    geometric_acceptance_pdf,
)
from src.util import Stage, get_cmes_full_model_from_pickle, get_experiment_id

DRAFT_ENERGY_MULTIPLIER = 0.1  # Example multiplier for draft model
DRAFT_LATENCY_MULTIPLIER = 0.1  # Example multiplier for draft model

CONTEXT_LEN = 8
DECODE_LEN = 8
OUT_PREFIX = f"outputs/context_{CONTEXT_LEN}_decode_{DECODE_LEN}/"
if DECODE_LEN == 8:
    K_RANGE = [0, 1, 2, 3, 4, 5, 6, 7]
elif DECODE_LEN == 256:
    K_RANGE = [0, 1, 63, 127, 127 + 64, 255]
else:
    raise ValueError("Unsupported DECODE_LEN value")


def scale_results(results, model, stage):
    """Scale results based on the model and stage."""
    multiplier = 1 if stage == Stage.PREFILL else model.decode_size
    scaled_results = {k: v * multiplier for k, v in results.items()}
    return scaled_results


def get_cmes(config):
    """Get CMEs (Cost Model Evaluations) from the configuration."""
    experiment_name = get_experiment_name(config)
    out_path = config.out_path
    model = config.model
    stage = config.stage
    # Load CMEs from pickle file
    cmes_path = f"{out_path}/{experiment_name}/cmes.pickle"
    cmes = get_cmes_full_model_from_pickle(cmes_path, model, stage)
    return cmes


def sum_cme_attribute(cmes_list, attribute):
    """Sum a given attribute across a list of CMEs."""
    return sum(getattr(cme, attribute) for cme in cmes_list)


def get_results(config, results_name):
    """Get results json."""
    experiment_name = get_experiment_name(config)
    out_path = config.out_path
    model = config.model
    stage = config.stage
    # Load results from JSON file
    results_path = f"{out_path}/{experiment_name}/overall_simple.json"
    with open(results_path, "r") as f:
        results = json.load(f)
    results_scaled = scale_results(results, model, stage)
    return SimpleResults(results_name, results_scaled)


def get_experiment_name(config):
    model = config.model
    stage = config.stage
    quant = config.quant
    accelerator = config.accelerator
    experiment_name = get_experiment_id(model, stage, quant, accelerator)
    return experiment_name


def get_fallback_results(fallback_config, k):
    """Get fallback results for a specific k."""
    if k == DECODE_LEN:
        # No fallback needed if k == DECODE_LEN
        results_fallback = {"energy": 0, "latency": 0}
        return SimpleResults(
            name=f"speculative_fallback_k_{k}", results=results_fallback
        )
    else:
        fallback_config.model.prefill_size = CONTEXT_LEN + k
        fallback_config.model.decode_size = DECODE_LEN - k
        return get_results(
            fallback_config,
            results_name=f"speculative_fallback_k_{k}",
        )


if __name__ == "__main__":
    ## STANDARD PREFILL AND DECODE
    # prefill
    standard_config_prefill = get_standard_decoding_config_prefill(
        context_len=CONTEXT_LEN, out_prefix=OUT_PREFIX
    )
    results_standard_prefill = get_results(
        standard_config_prefill, results_name="standard_prefill"
    )
    # decode
    standard_config_decode = get_standard_decoding_config_decode(
        context_len=CONTEXT_LEN, decode_len=DECODE_LEN, out_prefix=OUT_PREFIX
    )
    results_standard_decode = get_results(
        standard_config_decode, results_name="standard_decode"
    )
    total_results_standard = results_standard_prefill + results_standard_decode
    total_results_standard.pprint()

    ## SPECULATIVE STAGES
    # verification
    verification_config = get_speculative_target_verification_config(
        context_len=CONTEXT_LEN, out_prefix=OUT_PREFIX
    )
    results_verification = get_results(
        verification_config, results_name="speculative_verification"
    )
    # draft
    draft_config = get_speculative_draft_config(
        context_len=CONTEXT_LEN, decode_len=DECODE_LEN, out_prefix=OUT_PREFIX
    )
    results_draft = get_results(draft_config, results_name="speculative_draft")
    # fallback
    all_speculative_results = []
    for k in K_RANGE:
        context_len = CONTEXT_LEN + k
        decode_len = DECODE_LEN - k
        fallback_config = get_speculative_target_fallback_config(
            context_len=context_len, decode_len=decode_len, out_prefix=OUT_PREFIX
        )
        results_fallback = get_fallback_results(fallback_config, k)
        total_results_speculative = (
            results_verification + results_draft + results_fallback
        )
        all_speculative_results.append(total_results_speculative)
        # total_results_speculative.pprint()

    # Get the probability for each k to occur
    geo_probs = geometric_acceptance_pdf(p=0.8, draft_len=DECODE_LEN)
    exp_probs = exp_decay_acceptance_pdf(draft_len=DECODE_LEN, p0=0.8, lam=0.2)
    print("Geometric Acceptance Probabilities:", geo_probs)
    print("Exponential Decay Acceptance Probabilities:", exp_probs)
    # avg_energy_geo = sum(
    #     result.energy * prob for result, prob in zip(all_speculative_results, geo_probs)
    # )
    # avg_latency_geo = sum(
    #     result.latency * prob
    #     for result, prob in zip(all_speculative_results, geo_probs)
    # )
    # avg_energy_exp = sum(
    #     result.energy * prob for result, prob in zip(all_speculative_results, exp_probs)
    # )
    # avg_latency_exp = sum(
    #     result.latency * prob
    #     for result, prob in zip(all_speculative_results, exp_probs)
    # )
    # print("Average Energy (Geometric):", f"{avg_energy_geo:.2e}")
    # print("Average Latency (Geometric):", f"{avg_latency_geo:.2e}")
    # print("Average Energy (Exponential):", f"{avg_energy_exp:.2e}")
    # print("Average Latency (Exponential):", f"{avg_latency_exp:.2e}")

    # srp = SimpleResultsPlotter(all_speculative_results, CONTEXT_LEN, DECODE_LEN)
    # srp.plot(fig_path=f"{OUT_PREFIX}/speculative_results_plot.png")

    # Plot results using cmes for the different standard stages
    cmes_standard_prefill = get_cmes(standard_config_prefill)
    cmes_standard_decode = get_cmes(standard_config_decode)
    plot_energy_and_latency(
        cmes_all=[cmes_standard_prefill, cmes_standard_decode],
        supergroups=["Standard Prefill", "Standard Decode"],
        title="Standard Decoding",
        filename=f"{OUT_PREFIX}/standard_cmes_plot.png",
    )

    standard_energy_total_prefill = sum_cme_attribute(
        cmes_standard_prefill, "energy_total"
    )
    standard_energy_total_decode = sum_cme_attribute(
        cmes_standard_decode, "energy_total"
    )
    standard_energy_total = standard_energy_total_prefill + standard_energy_total_decode

    standard_latency_total_prefill = sum_cme_attribute(
        cmes_standard_prefill, "latency_total2"
    )
    standard_latency_total_decode = sum_cme_attribute(
        cmes_standard_decode, "latency_total2"
    )
    standard_latency_total = (
        standard_latency_total_prefill + standard_latency_total_decode
    )
    print(f"Standard Energy Total: {standard_energy_total:.2e}")
    print(f"Standard Latency Total: {standard_latency_total:.2e}")

    # Plot results using cmes for the different speculative stages
    cmes_speculative_draft = get_cmes(draft_config)
    speculative_draft_energy_total = sum_cme_attribute(
        cmes_speculative_draft, "energy_total"
    )
    speculative_draft_latency_total = sum_cme_attribute(
        cmes_speculative_draft, "latency_total2"
    )
    cmes_speculative_verification = get_cmes(verification_config)
    speculative_verification_energy_total = sum_cme_attribute(
        cmes_speculative_verification, "energy_total"
    )
    speculative_verification_latency_total = sum_cme_attribute(
        cmes_speculative_verification, "latency_total2"
    )
    scaled_energies_geo = []
    scaled_latencies_geo = []
    scaled_energies_exp = []
    scaled_latencies_exp = []
    # Plot energy and latency for each k from 0 to DECODE_LEN
    for k in K_RANGE:
        cmes_speculative_fallback = get_cmes(
            get_speculative_target_fallback_config(
                context_len=CONTEXT_LEN + k,
                decode_len=DECODE_LEN - k,
                out_prefix=OUT_PREFIX,
            )
        )
        plot_energy_and_latency(
            cmes_all=[
                cmes_speculative_draft,
                cmes_speculative_verification,
                cmes_speculative_fallback,
            ],
            supergroups=["Draft", "Verification", "Fallback"],
            title=f"Speculative Decoding with k={k}",
            filename=f"{OUT_PREFIX}/speculative_cmes_plot_k_{k}.png",
        )
        speculative_fallback_energy_total = sum_cme_attribute(
            cmes_speculative_fallback, "energy_total"
        )
        speculative_fallback_latency_total = sum_cme_attribute(
            cmes_speculative_fallback, "latency_total2"
        )
        speculative_energy_total = (
            speculative_draft_energy_total
            + speculative_verification_energy_total
            + speculative_fallback_energy_total
        )
        speculative_latency_total = (
            speculative_draft_latency_total
            + speculative_verification_latency_total
            + speculative_fallback_latency_total
        )
        print(f"Speculative Energy Total (k={k}): {speculative_energy_total:.2e}")
        print(f"Speculative Latency Total (k={k}): {speculative_latency_total:.2e}")
        scaled_energies_geo.append(speculative_fallback_energy_total * geo_probs[k])
        scaled_latencies_geo.append(speculative_fallback_latency_total * geo_probs[k])
        scaled_energies_exp.append(speculative_fallback_energy_total * exp_probs[k])
        scaled_latencies_exp.append(speculative_fallback_latency_total * exp_probs[k])

    # Print the total average energy and latency by summing the scaled values and adding the draft and verification contributions
    avg_energy_geo = (
        speculative_draft_energy_total
        + speculative_verification_energy_total
        + sum(scaled_energies_geo)
    )
    avg_latency_geo = (
        speculative_draft_latency_total
        + speculative_verification_latency_total
        + sum(scaled_latencies_geo)
    )
    avg_energy_exp = (
        speculative_draft_energy_total
        + speculative_verification_energy_total
        + sum(scaled_energies_exp)
    )
    avg_latency_exp = (
        speculative_draft_latency_total
        + speculative_verification_latency_total
        + sum(scaled_latencies_exp)
    )
    print("Average Energy (Geometric):", f"{avg_energy_geo:.2e}")
    print("Average Latency (Geometric):", f"{avg_latency_geo:.2e}")
    print("Average Energy (Exponential):", f"{avg_energy_exp:.2e}")
    print("Average Latency (Exponential):", f"{avg_latency_exp:.2e}")
