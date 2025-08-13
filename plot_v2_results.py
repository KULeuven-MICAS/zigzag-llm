import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def get_directories(path):
    return [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]

def load_sample_data(out_prefix):
    """Load energy and latency data from sample JSON files."""
    sample_data = []
    for sample_folder in sorted(get_directories(out_prefix), key=lambda x: int(x.split('_')[-1])):
        sample_path = os.path.join(out_prefix, sample_folder)
        sample_file = os.path.join(sample_path, "sample_performance.json")
        if os.path.isfile(sample_file):
            with open(sample_file, "r") as f:
                data = json.load(f)
                sample_data.append({
                    "sample": int(sample_folder.split('_')[-1]),
                    "energy": data["energy"],
                    "latency": data["latency"]
                })
    return sample_data

def load_standard_data(standard_path):
    """Load energy and latency data from the standard performance JSON file."""
    standard_file = os.path.join(standard_path, "performance.json")
    if os.path.isfile(standard_file):
        with open(standard_file, "r") as f:
            data = json.load(f)
            return {
                "sample": "Standard",
                "energy": data["energy"],
                "latency": data["latency"]
            }
    return None

def plot_energy_and_latency_line(sample_data, standard_data, out_prefix, include_standard):
    """Create line plots for energy and latency across samples."""
    sample_data = sorted(sample_data, key=lambda x: x["sample"])
    samples = [d["sample"] for d in sample_data]
    energies = [d["energy"] for d in sample_data]
    latencies = [d["latency"] for d in sample_data]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Energy line plot
    sns.lineplot(x=samples, y=energies, marker="o", ax=axes[0], color="skyblue", linewidth=2, label="Speculative")
    if include_standard and standard_data:
        axes[0].scatter([standard_data["sample"]], [standard_data["energy"]], color="red", marker="s", s=100, label="Standard")
    axes[0].set_title("Energy Across Samples", fontsize=14)
    axes[0].set_ylabel("Energy", fontsize=12)
    axes[0].tick_params(axis='y', labelsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend()

    # Latency line plot
    sns.lineplot(x=samples, y=latencies, marker="o", ax=axes[1], color="lightgreen", linewidth=2, label="Speculative")
    if include_standard and standard_data:
        axes[1].scatter([standard_data["sample"]], [standard_data["latency"]], color="red", marker="s", s=100, label="Standard")
    axes[1].set_title("Latency Across Samples", fontsize=14)
    axes[1].set_ylabel("Latency", fontsize=12)
    axes[1].set_xlabel("Samples", fontsize=12)
    axes[1].tick_params(axis='y', labelsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.xticks(samples[::max(1, len(samples)//10)], fontsize=10)  # Show a subset of sample numbers
    plt.tight_layout()
    suffix = "_with_standard" if include_standard else "_without_standard"
    line_plot_path = os.path.join(out_prefix, f"line_plots_energy_latency{suffix}.png")
    plt.savefig(line_plot_path, dpi=300)
    print(f"Line plots saved to {line_plot_path}")
    plt.close()

def plot_energy_and_latency_box(sample_data, standard_data, out_prefix, include_standard):
    """Create box plots for energy and latency distributions across samples."""
    energies = [d["energy"] for d in sample_data]
    latencies = [d["latency"] for d in sample_data]

    fig, axes = plt.subplots(2, 1, figsize=(6, 10), sharex=False)

    # Energy box plot
    sns.boxplot(y=energies, ax=axes[0], color="skyblue", width=0.4, linewidth=1.5)
    avg_energy = sum(energies) / len(energies)
    axes[0].axhline(avg_energy, color="blue", linestyle="--", linewidth=1.5, label=f"Average: {avg_energy:.2e}")
    if include_standard and standard_data:
        axes[0].scatter([0], [standard_data["energy"]], color="red", marker="s", s=100, label=f"Standard: {standard_data['energy']:.2e}")
    axes[0].set_title("Energy Distribution Across Samples", fontsize=14)
    axes[0].set_ylabel("Energy", fontsize=12)
    axes[0].tick_params(axis='y', labelsize=10)
    axes[0].legend(fontsize=12)

    # Latency box plot
    sns.boxplot(y=latencies, ax=axes[1], color="lightgreen", width=0.4, linewidth=1.5)
    avg_latency = sum(latencies) / len(latencies)
    axes[1].axhline(avg_latency, color="green", linestyle="--", linewidth=1.5, label=f"Average: {avg_latency:.2e}")
    if include_standard and standard_data:
        axes[1].scatter([0], [standard_data["latency"]], color="red", marker="s", s=100, label=f"Standard: {standard_data['latency']:.2e}")
    axes[1].set_title("Latency Distribution Across Samples", fontsize=14)
    axes[1].set_ylabel("Latency", fontsize=12)
    axes[1].tick_params(axis='y', labelsize=10)
    axes[1].legend(fontsize=12)

    # Add gridlines for better readability
    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    suffix = "_with_standard" if include_standard else "_without_standard"
    box_plot_path = os.path.join(out_prefix, f"box_plots_energy_latency{suffix}.png")
    plt.savefig(box_plot_path, dpi=300)
    print(f"Box plots saved to {box_plot_path}")
    plt.close()

def main():
    CONTEXT_LEN = 256
    DECODE_LEN = 256
    DRAFT_DECODE_LEN = 5
    OUT_PREFIX = f"outputs/v2/initial_context_{CONTEXT_LEN}_to_decode_{DECODE_LEN}_total_with_{DRAFT_DECODE_LEN}_draft_decode/"
    STANDARD_PATH = f"outputs/v2/standard/context_{CONTEXT_LEN}_decode_{DECODE_LEN}/standard/"

    sample_data = load_sample_data(OUT_PREFIX)
    standard_data = load_standard_data(STANDARD_PATH)

    if not sample_data:
        print("No sample data found. Ensure the parse script has been run.")
        return

    include_standard = True  # Set this flag to include or exclude the standard point

    plot_energy_and_latency_line(sample_data, standard_data, OUT_PREFIX, include_standard)
    plot_energy_and_latency_box(sample_data, standard_data, OUT_PREFIX, include_standard)

if __name__ == "__main__":
    main()
