from src.plots import plot_energy_and_latency
from src.util import (
    get_cmes_full_model_from_pickle,
    get_experiment_id,
)


def visualize_experiment(experiment_config, supergroups, out_path, fig_title, fig_path):
    model = experiment_config.model
    stage = experiment_config.stage
    quant = experiment_config.quant
    accelerator = experiment_config.accelerator
    experiment_name = get_experiment_id(model, stage, quant, accelerator)
    pickle_filename = f"{out_path}/{experiment_name}/cmes.pickle"
    cmes_full_model = get_cmes_full_model_from_pickle(pickle_filename, model, stage)

    plot_energy_and_latency(
        [cmes_full_model],
        supergroups=supergroups,
        title=fig_title,
        filename=fig_path,
    )
