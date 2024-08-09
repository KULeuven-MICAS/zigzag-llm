import itertools
from time import sleep
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sympy import true

from src.util import ARRAY_T, CME_T, GROUPS, LAYERS_TO_PLOT


def group_results(data: list[ARRAY_T]) -> ARRAY_T:
    """Here, we group the data of the CMEs in `LAYERS_TO_PLOT` to `GROUPS`
    Shape in: (5, len(bars), len(sections))
    Shape out: (3, len(bars), len(sections))"""
    assert len(data) == len(LAYERS_TO_PLOT)
    assert len(data[0].shape) == 2
    return np.array([data[0], data[1] + data[2], data[3] + data[4]])


def group_results_single_bar(data: list[ARRAY_T]) -> ARRAY_T:
    """Here, we group the data of the CMEs in `LAYERS_TO_PLOT` to `GROUPS`
    Shape in: (5, len(sections))
    Shape out: (3, len(sections))"""
    assert len(data) == len(LAYERS_TO_PLOT)
    assert len(data[0].shape) == 1
    return np.array([data[0], data[1] + data[2], data[3] + data[4]])


class PlotCMEMinimal:
    bars = GROUPS
    energy_sections = ["MAC", "RF", "SRAM", "DRAM"]

    @staticmethod
    def cme_to_energy_array_single_bar(cme: CME_T) -> ARRAY_T:
        """Energy per memory, summed up for all operands"""
        operands = ["W", "I", "O"]
        data = cme.__jsonrepr__()["outputs"]["energy"]
        result = [data["operational_energy"]]
        result += [
            np.sum([PlotCMEDetailed.get_mem_energy(data, op, mem_level) for op in operands]) for mem_level in range(3)
        ]
        return np.array(result)

    @staticmethod
    def cmes_to_energy_array_single_group(cmes: list[CME_T]) -> ARRAY_T:
        return group_results_single_bar([PlotCMEMinimal.cme_to_energy_array_single_bar(cme) for cme in cmes])

    @staticmethod
    def cmes_to_array_single_group(cmes: list[CME_T]) -> ARRAY_T:
        return group_results_single_bar([PlotCMEDetailed.cme_to_latency_array_single_bar(cme) for cme in cmes])


class PlotCMEDetailed:
    energy_bars = ["MAC", "RF", "SRAM", "DRAM"]
    energy_sections = ["MAC", "weight", "act", "act2", "output"]
    non_weight_layers = [1, 2]  # Indices in `LAYERS_TO_PLOT`

    latency_sections = ["Ideal computation", "Spatial stall", "Memory stall"]

    @staticmethod
    def get_mem_energy(data: Any, op: str, mem_level: int):
        # There should be 3 mem levels. Insert 0 at lowest level otherwise
        energy_per_level = data["memory_energy_breakdown_per_level"][op]
        if len(energy_per_level) == 2:
            energy_per_level = [0] + energy_per_level
        return energy_per_level[mem_level]

    @staticmethod
    def cme_to_energy_array_single_group(cme: CME_T, is_weight_layer: bool = True):
        """Energy per memory, per operand. This will return a single group"""

        operands = ["W", "I", "O"]  # Same order as `sections`
        data = cme.__jsonrepr__()["outputs"]["energy"]
        result = np.zeros((len(PlotCMEDetailed.energy_bars), len(PlotCMEDetailed.energy_sections)))
        result[0] = [data["operational_energy"]] + (len(PlotCMEDetailed.energy_sections) - 1) * [0]
        for mem_level, _ in enumerate(PlotCMEDetailed.energy_bars[1:]):
            energy_per_op = [PlotCMEDetailed.get_mem_energy(data, op, mem_level) for op in operands]
            if is_weight_layer:
                # Put the energy for `W` at label `weight`, set `act2` to 0
                result[mem_level + 1] = [0] + energy_per_op[:2] + [0, energy_per_op[2]]
            else:
                # Put the energy for `W` at label `act2`, set `weight` to 0
                result[mem_level + 1] = [0, 0, energy_per_op[1], energy_per_op[0], energy_per_op[2]]

        return result

    @staticmethod
    def cmes_to_energy_array_all(cmes: list[CME_T]):
        return group_results(
            [
                PlotCMEDetailed.cme_to_energy_array_single_group(
                    cme, is_weight_layer=idx not in PlotCMEDetailed.non_weight_layers
                )
                for idx, cme in enumerate(cmes)
            ]
        )

    @staticmethod
    def cme_to_latency_array_single_bar(cme: CME_T):
        """Latency per category.
        Shape = (len(sections))"""
        # Hard-copied from zigzag `plot_cme`
        result = np.array(
            [
                cme.ideal_cycle,  # Ideal computation
                (cme.ideal_temporal_cycle - cme.ideal_cycle),  # Spatial stall
                (cme.latency_total0 - cme.ideal_temporal_cycle)  # Temporal stall
                + (cme.latency_total1 - cme.latency_total0)  # Data loading
                + (cme.latency_total2 - cme.latency_total1),  # Data off-loading
            ]
        )
        return result

    @staticmethod
    def cme_to_latency_array_single_group(cme: CME_T):
        # Single bar per group
        return np.array([PlotCMEDetailed.cme_to_latency_array_single_bar(cme)])

    @staticmethod
    def cme_to_latency_array_all_single_group(cmes: list[CME_T]):
        assert len(cmes) == len(LAYERS_TO_PLOT)
        # All CMEs in single group
        return group_results_single_bar([PlotCMEDetailed.cme_to_latency_array_single_bar(cme) for cme in cmes])

    @staticmethod
    def cme_to_latency_array_all(cmes: list[CME_T]):
        return group_results([PlotCMEDetailed.cme_to_latency_array_single_group(cme) for cme in cmes])


class BarPlotter:
    def __init__(
        self,
        groups: list[str],
        bars: list[str],
        sections: list[str],
        *,
        supergroups: list[str] | None = None,
        # Layout
        bar_width: float = 0.6,
        bar_spacing: float = 0.1,
        group_spacing: float = 1,
        group_name_dy: float = -4,
        group_name_offset: float | None = None,
        scale: str = "linear",
        ylim: float | None = None,
        # Labels
        group_name_fontsize: int = 14,
        group_name_color: str = "black",
        xtick_labels: list[str] | None = None,
        xtick_rotation: int = 45,
        xtick_fontsize: int = 14,
        xtick_ha: str = "right",
        ylabel: str = "",
        title: str = "",
        legend_cols: int = 1,
        # Other
        colors: None = None,
    ):
        assert xtick_labels is None or len(xtick_labels) == len(groups) * len(bars)
        self.groups = groups
        self.bars = bars
        self.sections = sections
        self.supergroups = supergroups
        # Layout
        self.bar_width = bar_width
        self.bar_spacing = bar_spacing
        self.group_spacing = group_spacing
        self.group_name_dy = group_name_dy
        self.group_name_offset = (
            (len(self.bars) * (self.bar_width + self.bar_spacing)) * 0.4
            if group_name_offset is None
            else group_name_offset
        )
        self.scale = scale
        self.ylim = ylim

        # Labels
        self.group_name_fontsize = group_name_fontsize
        self.group_name_color = group_name_color
        self.xtick_labels = xtick_labels if xtick_labels is not None else len(groups) * bars
        self.xtick_rotation = xtick_rotation
        self.xtick_fontsize = xtick_fontsize
        self.xtick_ha = xtick_ha
        # Offset from bar center
        self.xtick_offset = self.bar_width / 2 if xtick_ha == "right" else 0
        self.ylabel = ylabel
        self.title = title
        self.legend_cols = legend_cols

        # Other
        colors_default = seaborn.color_palette("pastel", len(self.sections))
        colors_default = colors_default[2:] + colors_default[:2]  # Because green is at idx 2
        self.colors = colors_default if colors is None else colors
        # Use this setting to saturate the bars and print their actual value on top
        self.nb_saturated_bars = 0

    def construct_subplot(self, ax: Any, data: ARRAY_T):
        assert data.shape == (len(self.groups), len(self.bars), len(self.sections))

        indices = np.arange(len(self.groups)) * (
            len(self.bars) * (self.bar_width + self.bar_spacing) + self.group_spacing
        )
        group_name_positions = indices + self.group_name_offset

        # Saturate data
        if self.nb_saturated_bars > 0:
            data, labels_on_top = self.saturate_data(data, nb_saturated_bars=self.nb_saturated_bars)

        # Make bars
        for i, _ in enumerate(self.bars):
            bottom = np.zeros(len(self.groups))
            positions = indices + i * (self.bar_width + self.bar_spacing)

            for j, section in enumerate(self.sections):
                heights = data[:, i, j]
                ax.bar(
                    positions,
                    heights,
                    self.bar_width,
                    bottom=bottom,
                    label=f"{section}" if i == 0 else "",
                    color=self.colors[j],
                    edgecolor="black",
                )
                bottom += heights

        # Labels on top of saturated bars
        if self.nb_saturated_bars > 0:
            saturated_value = data.max()
            for i in range(len(self.groups)):
                for j in range(len(self.bars)):
                    label = labels_on_top[i][j]
                    if label != 0:
                        ax.text(
                            indices[i] + j * (self.bar_width + self.bar_spacing),
                            saturated_value,
                            f"{label:.1e}",
                            bbox=dict(facecolor="black"),
                            color="white",
                            ha="center",
                        )

        # Bar names (as xticks)
        xticks_positions = [
            indices[i] + j * (self.bar_width + self.bar_spacing) + self.xtick_offset
            for i in range(len(self.groups))
            for j in range(len(self.bars))
        ]
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(
            self.xtick_labels, fontsize=self.xtick_fontsize, ha=self.xtick_ha, rotation=self.xtick_rotation
        )

        # Group names
        for i, group in enumerate(self.groups):
            ax.annotate(
                group,
                xy=(group_name_positions[i], 0),  # Reference in coordinate system
                xycoords="data",  # Use coordinate system of data points
                xytext=(0, self.group_name_dy),  # Offset from reference
                textcoords="offset fontsize",  # Offset value is relative to fontsize
                ha="center",
                va="top",
                weight="normal",
                fontsize=self.group_name_fontsize,
                rotation=0,
                color=self.group_name_color,
            )

        if self.supergroups is not None:
            groups_per_supergroup = len(self.groups) // len(self.supergroups)
            # e.g. 3 groups in each supergroup -> put supergroup name under group name at idx=1 within the supergroup
            supergroup_idx_within_group = groups_per_supergroup // len(self.supergroups)
            # Supergroup names
            for i, supergroup in enumerate(self.supergroups):
                x_coordiante = group_name_positions[i * groups_per_supergroup + supergroup_idx_within_group]
                ax.annotate(
                    supergroup,
                    xy=(x_coordiante, 0),  # Reference in coordinate system
                    xycoords="data",  # Use coordinate system of data points
                    xytext=(0, self.group_name_dy - 0.9),  # Offset from reference
                    textcoords="offset fontsize",  # Offset value is relative to fontsize
                    ha="center",
                    va="top",
                    weight="normal",
                    fontsize=14,
                    rotation=0,
                )

        # Add labels and title
        # ax.set_xlabel("Module", fontsize=16)
        if self.ylim is not None:
            ax.set_ylim([0, self.ylim])
        ax.set_ylabel(self.ylabel, fontsize=16)
        ax.set_title(self.title, fontsize=16)
        ax.legend(ncol=self.legend_cols, fontsize=14, loc="upper left")

    def construct_subplot_broken_axis(self, ax: Any, data: ARRAY_T):
        assert data.shape == (len(self.groups), len(self.bars), len(self.sections))

        indices = np.arange(len(self.groups)) * (
            len(self.bars) * (self.bar_width + self.bar_spacing) + self.group_spacing
        )

        # Make bars
        for i, _ in enumerate(self.bars):
            bottom = np.zeros(len(self.groups))
            for j, _ in enumerate(self.sections):
                positions = indices + i * (self.bar_width + self.bar_spacing)
                heights = data[:, i, j]
                ax.bar(
                    positions,
                    heights,
                    self.bar_width,
                    bottom=bottom,
                    color=self.colors[j],
                    edgecolor="black",
                )
                bottom += heights

    def saturate_data(self, data: ARRAY_T, nb_saturated_bars: int = 0):
        original_values = np.zeros(data.shape[0:-1])

        altered_at_idx: list[tuple[np.intp, ...]] = []

        for _ in range(nb_saturated_bars):
            idx_max = np.unravel_index(np.argmax(data, axis=None), data.shape)
            altered_at_idx.append(idx_max)
            # Store original
            value_max = data[idx_max]
            original_values[idx_max[0:-1]] = value_max
            # Temporarily set to 0
            data[idx_max] = 0

        # What is the largest bar that will not be saturated?
        idx_max = np.unravel_index(np.argmax(data, axis=None), data.shape)
        highest_value = data[idx_max]
        saturated_value = highest_value * 1.5

        # Restore the values previously set to 0 to the saturated value
        for idx in altered_at_idx:
            data[idx] = saturated_value

        return data, original_values

    def plot(self, data: ARRAY_T, filename: str) -> None:
        plt.style.use("ggplot")
        plt.rc("font", family="DejaVu Serif")
        _, ax = plt.subplots(figsize=(12, 6))
        self.construct_subplot(ax, data)
        plt.yscale(self.scale)
        plt.tight_layout()
        plt.savefig(filename, transparent=False)


class BarPlotterSubfigures:
    def __init__(
        self,
        bar_plotters: list[BarPlotter],
        *,
        subplot_rows: int = 1,
        subplot_cols: int = 1,
        width_ratios: list[int | float] | None = None,
        title: str = "",
    ):
        self.nb_plots = subplot_rows * subplot_cols
        assert width_ratios is None or len(width_ratios) == subplot_cols
        assert len(bar_plotters) == self.nb_plots
        self.bar_plotters = bar_plotters
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols
        self.width_ratios = width_ratios if width_ratios is not None else subplot_cols * [1]
        self.title = title

    def plot_broken_axes(self, data: list[ARRAY_T], filename: str) -> None:
        assert len(data) == self.nb_plots

        fig, axes = plt.subplots(
            nrows=2 * self.subplot_rows,
            ncols=self.subplot_cols,
            width_ratios=self.width_ratios,
            figsize=(12, 6),
            sharex=true,
        )
        fig.subplots_adjust(hspace=0.001)

        for idx in range(self.nb_plots):
            ax1 = axes[0][idx]
            ax2 = axes[1][idx]

            # ax1.set_ylim(10**13, 2 * 10**13)
            # ax2.set_ylim(0, 10**12)
            ax1.spines.bottom.set_visible(False)
            ax2.spines.top.set_visible(False)
            ax1.tick_params(labeltop=False)

            d = 0.5  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(
                marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False
            )
            ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
            ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

            self.bar_plotters[idx].construct_subplot(ax2, data[idx])
            self.bar_plotters[idx].construct_subplot_broken_axis(ax1, data[idx])

        fig.suptitle(self.title)
        # plt.rc("font", family="DejaVu Serif")
        # plt.style.use("ggplot")
        plt.tight_layout()
        plt.savefig(filename, transparent=False)

    def plot(self, data: list[ARRAY_T], filename: str) -> None:
        assert len(data) == self.nb_plots

        plt.style.use("ggplot")
        fig, axises = plt.subplots(
            nrows=self.subplot_rows, ncols=self.subplot_cols, width_ratios=self.width_ratios, figsize=(12, 6)
        )

        for ax, data_subplot, plotter in zip(axises, data, self.bar_plotters):
            plotter.construct_subplot(ax, data_subplot)

        fig.suptitle(self.title)
        # plt.rc("font", family="DejaVu Serif")
        plt.tight_layout(pad=0.5, w_pad=0)
        plt.savefig(filename, transparent=False)
