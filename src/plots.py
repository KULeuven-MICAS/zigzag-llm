import itertools
import numpy as np

from src.util import ARRAY_T, CME_T, GROUPS
from src.plot_util import (
    BarPlotter,
    BarPlotterSubfigures,
    PlotCMEDetailed,
    PlotCMEMinimal,
    group_results,
)


def plot_energy_clean(cmes: list[CME_T], filename: str):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert len(cmes) == 5, "Are these the CMEs from `LAYERS_TO_PLOT`?"

    bars = PlotCMEDetailed.energy_bars  # ["MAC", "RF", "SRAM", "DRAM"]
    sections = PlotCMEDetailed.energy_sections  # ["MAC", "weight", "act", "act2", "output"]

    data = PlotCMEDetailed.cmes_to_energy_array_all(cmes)
    p = BarPlotter(GROUPS, bars, sections, legend_cols=4, ylabel="Energy (pJ)")
    p.plot(data, filename)


def plot_energy_minimal(cmes: list[CME_T], filename: str):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert len(cmes) == 5, "Are these the CMEs from `LAYERS_TO_PLOT`?"

    bars = [""]
    sections = PlotCMEMinimal.energy_sections  # ["MAC", "RF", "SRAM", "DRAM"]

    def cme_to_energy_array_single_group(cme: CME_T):
        # Group has 1 bar
        return np.array([PlotCMEMinimal.cme_to_energy_array_single_bar(cme)])

    data = group_results([cme_to_energy_array_single_group(cme) for cme in cmes])
    p = BarPlotter(GROUPS, bars, sections, legend_cols=4, ylabel="Energy (pJ)", group_name_offset=0, group_name_dy=-1)
    p.plot(data, filename)


def plot_energy_compare(cmes_all: list[list[CME_T]], supergroups: list[str], filename: str, title: str = ""):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert all([len(cmes) == 5 for cmes in cmes_all]), "Are these the CMEs from `LAYERS_TO_PLOT`?"
    assert len(cmes_all) == len(supergroups)

    groups = [f"{group}\n{supergroup}" for supergroup, group in itertools.product(supergroups, GROUPS)]
    bars = PlotCMEDetailed.energy_bars
    sections = PlotCMEDetailed.energy_sections

    data: ARRAY_T = np.zeros((0, len(bars), len(sections)))
    for cmes_per_supergroup in cmes_all:
        data_per_supergroup = PlotCMEDetailed.cmes_to_energy_array_all(cmes_per_supergroup)
        data = np.concatenate([data, data_per_supergroup], axis=0)

    p = BarPlotter(groups, bars, sections, legend_cols=5, ylabel="Energy (pJ)", title=title)
    p.plot(data, filename)


def plot_energy_compare_minimal(cmes_all: list[list[CME_T]], groups: list[str], filename: str, title: str = ""):
    """1 bar corresponds to 1 layer, 1 group to a single zigzag simulation
    `cmes` correspond to `LAYERS_TO_PLOT`"""
    assert all([len(cmes) == 5 for cmes in cmes_all]), "Are these the CMEs from `LAYERS_TO_PLOT`?"
    assert len(cmes_all) == len(groups)

    bars = PlotCMEMinimal.bars
    sections = PlotCMEMinimal.energy_sections

    data = np.array(
        [PlotCMEMinimal.cmes_to_energy_array_single_group(cmes_single_group) for cmes_single_group in cmes_all]
    )

    p = BarPlotter(
        groups,
        bars,
        sections,
        group_spacing=0.5,
        legend_cols=5,
        group_name_dy=-5,
        ylabel="Energy (pJ)",
        title=title,
    )
    p.plot(data, filename)


def plot_latency_clean(cmes: list[CME_T], filename: str):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert len(cmes) == 5, "Are these the CMEs from `LAYERS_TO_PLOT`?"

    bars = [""]
    sections = PlotCMEDetailed.latency_sections

    data = group_results([PlotCMEDetailed.cme_to_latency_array_single_group(cme) for cme in cmes])
    p = BarPlotter(
        GROUPS,
        bars,
        sections,
        bar_width=1,
        bar_spacing=0,
        group_spacing=0.8,
        group_name_offset=0,
        group_name_dy=-1,
        ylabel="Latency (cycles)",
    )
    p.plot(data, filename)


def plot_latency_compare(
    cmes_all: list[list[CME_T]],
    supergroups: list[str],
    filename: str,
    title: str = "",
    ylim: float | None = None,
):
    """1 group corresponds to 1 layer (similar to `plot_energy_compare`)
    `cmes` correspond to `LAYERS_TO_PLOT`"""
    assert all([len(cmes) == 5 for cmes in cmes_all]), "Are these the CMEs from `LAYERS_TO_PLOT`?"
    assert len(cmes_all) == len(supergroups)

    groups = [f"{group}\n{supergroup}" for supergroup, group in itertools.product(supergroups, GROUPS)]
    bars = [""]
    sections = PlotCMEDetailed.latency_sections

    data: ARRAY_T = np.zeros((0, len(bars), len(sections)))
    for cmes_per_supergroup in cmes_all:
        data_per_supergroup = group_results(
            [PlotCMEDetailed.cme_to_latency_array_single_group(cme) for cme in cmes_per_supergroup]
        )
        data = np.concatenate([data, data_per_supergroup], axis=0)

    assert data.shape == (len(supergroups) * len(GROUPS), len(bars), len(sections))

    p = BarPlotter(
        groups,
        bars,
        sections,
        bar_width=0.8,
        bar_spacing=0,
        group_spacing=0.8,
        group_name_offset=0,
        group_name_dy=-1,
        ylabel="Latency (cycles)",
        title=title,
        ylim=ylim,
    )
    p.plot(data, filename)


def plot_latency_compare_minimal(cmes_all: list[list[CME_T]], groups: list[str], filename: str, title: str = ""):
    """1 group corresponds to single zigzag simulation
    `cmes` correspond to `LAYERS_TO_PLOT`"""
    assert all([len(cmes) == 5 for cmes in cmes_all]), "Are these the CMEs from `LAYERS_TO_PLOT`?"
    assert len(cmes_all) == len(groups)

    bars = GROUPS
    sections = PlotCMEDetailed.latency_sections

    data = np.array([PlotCMEMinimal.cmes_to_array_single_group(cmes_single_group) for cmes_single_group in cmes_all])

    p = BarPlotter(
        groups,
        bars,
        sections,
        group_spacing=0.5,
        group_name_dy=-5,
        ylabel="Latency (cycles)",
        title=title,
    )
    p.plot(data, filename)


def plot_energy_and_latency(
    cmes_all: list[list[CME_T]],
    supergroups: list[str],
    title: str,
    filename: str,
    ylim_energy: float | None = None,
    ylim_latency: float | None = None,
):
    assert all([len(cmes) == 5 for cmes in cmes_all]), "Are these the CMEs from `LAYERS_TO_PLOT`?"
    assert len(cmes_all) == len(supergroups)

    # energy_groups = [f"{group}\n{supergroup}" for supergroup, group in itertools.product(supergroups, GROUPS)]
    energy_groups = len(supergroups) * GROUPS
    energy_bars = PlotCMEDetailed.energy_bars
    energy_sections = PlotCMEDetailed.energy_sections

    latency_groups = len(supergroups) * [""]
    latency_bars = GROUPS
    latency_sections = PlotCMEDetailed.latency_sections

    energy_data: ARRAY_T = np.zeros((0, len(energy_bars), len(energy_sections)))

    for cmes_per_supergroup in cmes_all:
        energy_data_per_supergroup = PlotCMEDetailed.cmes_to_energy_array_all(cmes_per_supergroup)
        energy_data = np.concatenate([energy_data, energy_data_per_supergroup], axis=0)

    latency_data = np.array(
        [PlotCMEDetailed.cme_to_latency_array_all_single_group(cmes_per_supergroup) for cmes_per_supergroup in cmes_all]
    )

    energy_plotter = BarPlotter(
        energy_groups,
        energy_bars,
        energy_sections,
        supergroups=supergroups,
        bar_width=0.9,
        bar_spacing=0,
        group_spacing=0.7,
        group_name_dy=-4.8,
        group_name_fontsize=12,
        group_name_color="grey",
        xtick_rotation=90,
        ylabel="Energy (pJ)",
        ylim=ylim_energy,
    )
    latency_plotter = BarPlotter(
        latency_groups,
        latency_bars,
        latency_sections,
        supergroups=supergroups,
        bar_width=1.2,
        bar_spacing=0,
        group_spacing=0.8,
        group_name_dy=-4.8,
        ylabel="Latency (cycles)",
        ylim=ylim_latency,
    )

    p = BarPlotterSubfigures([energy_plotter, latency_plotter], subplot_cols=2, width_ratios=[3, 1], title=title)
    p.plot([energy_data, latency_data], filename)


def plot_energy_and_latency_minimal(cmes_all: list[list[CME_T]], groups: list[str], title: str, filename: str):
    assert all([len(cmes) == 5 for cmes in cmes_all]), "Are these the CMEs from `LAYERS_TO_PLOT`?"
    assert len(cmes_all) == len(groups)

    energy_bars = PlotCMEMinimal.bars
    energy_sections = PlotCMEMinimal.energy_sections
    latency_bars = GROUPS
    latency_sections = PlotCMEDetailed.latency_sections

    energy_data = np.array(
        [PlotCMEMinimal.cmes_to_energy_array_single_group(cmes_single_group) for cmes_single_group in cmes_all]
    )

    latency_data = np.array(
        [PlotCMEDetailed.cme_to_latency_array_all_single_group(cmes_single_group) for cmes_single_group in cmes_all]
    )

    energy_plotter = BarPlotter(
        groups,
        energy_bars,
        energy_sections,
        bar_width=0.9,
        bar_spacing=0,
        group_spacing=0.7,
        group_name_dy=-3.5,
        xtick_rotation=45,
        xtick_fontsize=8,
        ylabel="Energy (pJ)",
    )
    latency_plotter = BarPlotter(
        groups,
        latency_bars,
        latency_sections,
        bar_width=0.9,
        bar_spacing=0,
        group_spacing=0.7,
        xtick_fontsize=8,
        # xtick_labels=list(range(1, ))
        # group_name_offset=0,
        group_name_dy=-3.5,
        ylabel="Latency (cycles)",
    )

    p = BarPlotterSubfigures([energy_plotter, latency_plotter], subplot_cols=2, width_ratios=[1, 1], title=title)
    p.plot([energy_data, latency_data], filename)
