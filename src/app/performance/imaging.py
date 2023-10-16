from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from matplotlib import pyplot as plt
from io import BytesIO
import base64


def get_buffers(performance_results: dict) -> dict:
    model1 = performance_results[list(performance_results.keys())[0]]
    ds1 = model1[list(model1.keys())[0]]
    cuis = list(ds1["F1 for each CUI"].keys())

    # Create and save performance graphs for each dataset
    per_ds_figures = {}
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

    graph_buffers = {}  # To store the paths to the saved graphs
    for dataset_name, fig in per_ds_figures.items():
        plt.figure(fig)
        plt.xlabel("CUI")
        plt.xticks(x, cuis, rotation=90)  # Assuming keys are CUIs
        plt.ylabel("Score")
        plt.legend()
        plt.title(f"Performance Comparison for {dataset_name}")
        canvas = FigureCanvas(fig)
        buffer = BytesIO()
        canvas.print_png(buffer)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        graph_buffers[dataset_name] = plot_data
        plt.close()
    return graph_buffers
