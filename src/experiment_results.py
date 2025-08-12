import numpy as np

from src.plot_util import BarPlotter, BarPlotterSubfigures


class SimpleResults:
    def __init__(self, name, results):
        if not isinstance(results, dict):
            raise ValueError("results must be a dict")
        if "energy" not in results or "latency" not in results:
            raise KeyError("results must contain 'energy' and 'latency' keys")
        self.name = name
        self.energy = results["energy"]
        self.latency = results["latency"]

    def __add__(self, other):
        if not isinstance(other, SimpleResults):
            raise TypeError("Can only add SimpleResults instances")
        summed_results = {
            "energy": self.energy + other.energy,
            "latency": self.latency + other.latency,
        }
        return SimpleResults(f"{self.name}_plus_{other.name}", summed_results)

    def pprint(self):
        """Pretty print results."""
        print(f"Results for {self.name}:")
        print(f"energy: {self.energy:.2e}")
        print(f"latency: {self.latency:.2e}")
        print("\n")

    def get_energy(self):
        """Return the energy value."""
        return self.energy

    def get_latency(self):
        """Return the latency value."""
        return self.latency


class SimpleResultsPlotter:
    def __init__(self, results_list, context_len, decode_len):
        if not all(isinstance(r, SimpleResults) for r in results_list):
            raise TypeError("All items in results_list must be SimpleResults instances")
        self.results_list = results_list
        self.context_len = context_len
        self.decode_len = decode_len

    def plot(self, fig_path):
        # Prepare data for BarPlotter
        names = [r.name for r in self.results_list]
        energies = [r.get_energy() for r in self.results_list]
        latencies = [r.get_latency() for r in self.results_list]

        # BarPlotter expects a 3D array: (groups, bars, sections)
        # Here, we treat energy and latency as separate sections
        data = np.array(
            [[[energy, latency] for energy, latency in zip(energies, latencies)]]
        )

        # Create BarPlotter instance
        energy_plotter = BarPlotter(
            groups=["blabla"],
            bars=names,
            sections=["Energy (Joules)", "Latency (seconds)"],
            ylabel="Values",
            title="Experiment Results",
        )

        p = BarPlotterSubfigures(
            [energy_plotter, latency_plotter],
            subplot_cols=2,
            width_ratios=[3, 1],
            title=title,
        )
        p.plot([energy_data, latency_data], filename)
