from typing import Dict, Iterable, List, Tuple

# to set matplotlib backend
from . import imgutils  # noqa
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from matplotlib import pyplot as plt
from io import BytesIO
import base64

from ..medcat_linkage.medcat_integration import AllModelPerformanceResults

import logging

logger = logging.getLogger(__name__)


def _get_buffers_from_fig(title, x, cuis, fig, legend=[], ylabel="Score"):
    plt.figure(fig)
    plt.xlabel("CUI")
    plt.xticks(x, cuis, rotation=90)  # Assuming keys are CUIs
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.title(title)
    canvas = FigureCanvas(fig)
    buffer = BytesIO()
    canvas.print_png(buffer)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return plot_data


def _get_buffers(x: Iterable[int], cuis: List[str],
                 per_ds_figures: Dict[str, Figure]) -> Dict[str, str]:
    graph_buffers = {}  # To store the paths to the saved graphs
    for dataset_name, fig in per_ds_figures.items():
        title = f"Performance Comparison for {dataset_name}"
        model_plot = _get_buffers_from_fig(title, x, cuis, fig)
        graph_buffers[dataset_name] = model_plot
    return graph_buffers


def _plot_model_count_train(fig: Figure, model_data: Dict[str, int]
                            ) -> Tuple[Iterable[int], List[str]]:
    plt.figure(fig)
    cuis = list(model_data.keys())
    values = [model_data[cui] for cui in cuis]
    plt.plot(cuis, values)
    locs = range(len(cuis))
    return locs, cuis


def get_buffer_for_cui_count_train(data: Dict[str, Dict[str, int]],
                                   totals: List[int],
                                   ) -> str:
    fig = plt.figure()
    for model_data in data.values():
        locs, cuis = _plot_model_count_train(fig, model_data)
    legend = [f"{key} ({total})" for key, total in zip(data.keys(), totals)]
    return _get_buffers_from_fig("Count train", locs, cuis, fig,
                                 legend=legend, ylabel="Count")


def get_buffers(performance_results: AllModelPerformanceResults
                ) -> Dict[str, str]:
    model1 = performance_results[list(performance_results.keys())[0]]
    ds1 = model1[list(model1.keys())[0]]
    cuis = list(ds1["F1 for each CUI"].keys())
    x_vals = range(len(cuis))

    # Create and save performance graphs for each dataset
    per_ds_figures: Dict[str, Figure] = {}
    for model_name, model_performances in performance_results.items():
        for dataset_name, perf_data in model_performances.items():
            precision = perf_data.get("Precision for each CUI", {})
            recall = perf_data.get("Recall for each CUI", {})
            f1 = perf_data.get("F1 for each CUI", {})
            if dataset_name in per_ds_figures:
                fig = per_ds_figures[dataset_name]
                plt.figure(fig)
            else:
                fig = plt.figure(figsize=(10, 6))
                per_ds_figures[dataset_name] = fig
            x = range(len(precision))
            plt.plot(x, precision.values(), label=f"{model_name} Precision")
            plt.plot(x, recall.values(), label=f"{model_name} Recall")
            plt.plot(x, f1.values(), label=f"{model_name} F1")

    return _get_buffers(x_vals, cuis, per_ds_figures)
