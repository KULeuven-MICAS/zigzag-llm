from src.experiment_config import (
    get_standard_decoding_config_decode,
    get_standard_decoding_config_prefill,
)
from src.experiment_visualize import visualize_experiment

if __name__ == "__main__":
    # Visualize prefill
    standard_config_prefill = get_standard_decoding_config_prefill()
    supergroups = ["Prefill"]
    fig_title = f"Standard prefill ({standard_config_prefill.quant.name})"
    fig_path = f"{standard_config_prefill.out_path}/plot_prefill.png"
    visualize_experiment(
        standard_config_prefill,
        supergroups,
        fig_title=fig_title,
        out_path=standard_config_prefill.out_path,
        fig_path=fig_path,
    )
    standard_config_decode = get_standard_decoding_config_decode()
    supergroups = ["Decode"]
    fig_title = f"Standard Decode ({standard_config_decode.quant.name})"
    fig_path = f"{standard_config_decode.out_path}/plot_decode.png"
    visualize_experiment(
        standard_config_decode,
        supergroups,
        fig_title=fig_title,
        out_path=standard_config_decode.out_path,
        fig_path=fig_path,
    )
