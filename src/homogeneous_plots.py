"""Figure 5.2/5.3 comparisons from completed homogeneous benchmark runs.

Curves average independent seeds, not particle clouds pooled across runs.
Convergence panels use final-time errors versus n; trajectory panels use
max(n) for BKW and min(n), max(n) for the anisotropic example.
"""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_runs(paths):
    runs = []
    found = set()
    for path in map(Path, paths):
        configs = [path / "config.json"] if (path / "config.json").is_file() else sorted(path.rglob("config.json"))
        if not configs:
            raise ValueError(f"No runs found under {path}")
        for config_path in configs:
            root = config_path.parent.resolve()
            if root in found:
                continue
            found.add(root)
            # Incomplete runs are errors, not silently omitted from a sweep.
            if not (root / "summary.json").is_file():
                raise ValueError(f"Incomplete run: {root} (summary.json missing)")
            config = json.loads(config_path.read_text())
            if "example" not in config and "Example 5.1" in config.get("reference", ""):
                config["example"] = "bkw"  # compatibility with the initial implementation
            with (root / "metrics.csv").open() as handle:
                records = [{key: float(value) if value else None for key, value in row.items()}
                           for row in csv.DictReader(handle)]
            runs.append(dict(config=config, records=records,
                             summary=json.loads((root / "summary.json").read_text()), path=str(root)))
    validate_runs(runs)
    return runs


def validate_runs(runs):
    if not runs:
        raise ValueError("At least one completed run is required")
    first = runs[0]["config"]
    if first.get("example") not in ("bkw", "anisotropic"):
        raise ValueError("Expected a BKW or anisotropic homogeneous benchmark")
    # Training settings may differ between methods; physics/integration may not.
    common = ("example", "dv", "B", "dt", "t0", "final_time", "time_integrator", "fp32", "initial_variances")
    identities = set()
    training_config = None
    for run in runs:
        config = run["config"]
        if any(config.get(key) != first.get(key) for key in common):
            raise ValueError(f"Incompatible physics/time settings in {run['path']}; plot each configuration separately")
        if config["score_method"] == "sbtm":
            training = tuple(config.get(key) for key in
                             ("sbtm_training_stages", "sbtm_adaptive", "sbtm_num_batch_steps"))
            if training_config is not None and training != training_config:
                raise ValueError(f"Incompatible SBTM training settings in {run['path']}; plot each configuration separately")
            training_config = training
        identity = (config["score_method"], config["n"], config["seed"])
        if identity in identities:
            raise ValueError(f"Duplicate method/n/seed {identity}; select one run, not both retries")
        identities.add(identity)
        records = run["records"]
        if not records or not math.isclose(records[-1]["time"], config["final_time"], abs_tol=1e-12):
            raise ValueError(f"Final-time data missing in {run['path']}")
        if not math.isclose(records[0]["time"], config["t0"], abs_tol=1e-12):
            raise ValueError(f"Initial-time data missing in {run['path']}")
        if any(b["time"] <= a["time"] for a, b in zip(records, records[1:])):
            raise ValueError(f"Non-increasing diagnostic times in {run['path']}")


def trajectory(runs, name):
    times = np.array([row["time"] for row in runs[0]["records"]])
    data = []
    for run in runs:
        other = np.array([row["time"] for row in run["records"]])
        if other.shape != times.shape or not np.allclose(other, times, rtol=0, atol=1e-12):
            raise ValueError("Seed replicates need the same diagnostic time grid")
        values = np.array([row[name] for row in run["records"]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite or missing trajectory metric {name}")
        data.append(values)
    return times, np.mean(data, axis=0)


def convergence_data(runs, name):
    groups = defaultdict(list)
    for run in runs:
        value = run["summary"]["final"].get(name)
        if value is None or not math.isfinite(value):
            raise ValueError(f"Missing/nonfinite final {name} in {run['path']}")
        groups[(run["config"]["score_method"], run["config"]["n"])].append(value)
    rows = []
    for (method, n), values in sorted(groups.items()):
        rows.append(dict(metric=name, method=method, n=n, runs=len(values),
                         mean=float(np.mean(values)), std=float(np.std(values))))
    return rows


def plot_benchmarks(runs, output_dir):
    """Save the full paper-style panel layout, numerical CSV, and provenance."""
    validate_runs(runs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = runs[0]["config"]
    example = config["example"]
    methods = sorted({run["config"]["score_method"] for run in runs})
    ns = sorted({run["config"]["n"] for run in runs})
    groups = defaultdict(list)
    for run in runs:
        groups[(run["config"]["score_method"], run["config"]["n"])].append(run)
    colors = {"sbtm": "tab:blue", "blob": "tab:orange", "exact": "tab:green"}
    fig, axes = plt.subplots(3 if example == "bkw" else 2, 2,
                             figsize=(12, 12 if example == "bkw" else 8), squeeze=False)
    exported = []
    slopes = {}

    def time_panel(ax, metric, ylabel, selected_ns):
        for method in methods:
            for index, n in enumerate(selected_ns):
                if (method, n) not in groups:
                    continue
                times, values = trajectory(groups[(method, n)], metric)
                ax.plot(times, values, color=colors.get(method), linestyle="-" if index == 0 else ":",
                        label=f"{method}, n={n}")
        ax.set(xlabel="Physical time", ylabel=ylabel)

    def convergence_panel(ax, metric, ylabel):
        rows = convergence_data(runs, metric)
        exported.extend(rows)
        positives = [row["mean"] for row in rows if row["mean"] > 0]
        floor = min(positives) / 10 if positives else 1e-18
        for method in methods:
            selected = [row for row in rows if row["method"] == method]
            counts = np.array([row["n"] for row in selected])
            values = np.array([row["mean"] for row in selected])
            if np.any(values < 0):
                raise ValueError(f"Negative error in {metric}")
            positive = values > 0
            label = method
            if positive.sum() > 1:
                slope = float(np.polyfit(np.log(counts[positive]), np.log(values[positive]), 1)[0])
                slopes[f"{metric}/{method}"] = slope
                label += f", slope={slope:.2f}"
            ax.loglog(counts, np.maximum(values, floor), "o-", color=colors.get(method), label=label)
        if any(row["mean"] == 0 for row in rows):
            ax.text(0.03, 0.03, f"Exact zeros shown at {floor:.1e}", transform=ax.transAxes, fontsize=8)
        ax.set(xlabel="Number of particles, n", ylabel=ylabel)

    if example == "bkw":
        time_panel(axes[0, 0], "second_moment_11", r"$\Sigma_{11}$", [ns[-1]])
        axes[0, 0].axhline(1, color="black", linestyle="--", label="BKW")
        axes[0, 0].set_title(f"Covariance (1,1), n={ns[-1]}")
        convergence_panel(axes[0, 1], "second_moment_error_fro", r"$\|\Sigma(t_{end})-\Sigma^*(t_{end})\|_F$")
        axes[0, 1].set_title("Final second-moment error")
        time_panel(axes[1, 0], "score_relative_mse", "Normalized true score MSE", [ns[-1]])
        axes[1, 0].set_title(f"Score error, n={ns[-1]}")
        convergence_panel(axes[1, 1], "density_l2", r"$\|\widehat{u}(t_{end})-u^*(t_{end})\|_{L^2}$")
        axes[1, 1].set_title("Final reconstructed-density error")
        convergence_panel(axes[2, 0], "momentum_drift", r"$|\overline{v}(t_{end})-\overline{v}(t_0)|$")
        axes[2, 0].set_title("Sample mean drift")
        convergence_panel(axes[2, 1], "second_moment_trace_drift", r"$|\overline{|v|^2}(t_{end})-\overline{|v|^2}(t_0)|$")
        axes[2, 1].set_title("Sample second-moment trace drift")
        filename = "fig5_2"
    else:
        selected_ns = sorted({ns[0], ns[-1]})
        for i in range(2):
            time_panel(axes[0, i], f"second_moment_{i+1}{i+1}", rf"$\Sigma_{{{i+1}{i+1}}}$", selected_ns)
            times = np.linspace(config["t0"], config["final_time"], 401)
            initial = config["initial_variances"][i]
            equilibrium = np.mean(config["initial_variances"])
            reference = equilibrium + (initial - equilibrium) * np.exp(-4 * config["dv"] * config["B"] * times)
            axes[0, i].plot(times, reference, "k--", label="Analytical covariance")
            axes[0, i].set_title(f"Covariance ({i+1},{i+1})")
        convergence_panel(axes[1, 0], "second_moment_error_fro", r"$\|\Sigma(t_{end})-\Sigma^*(t_{end})\|_F$")
        axes[1, 0].set_title("Final second-moment error")
        time_panel(axes[1, 1], "estimated_entropy_rate", r"$\frac{1}{n}\sum_i s_i\cdot U_i$", selected_ns)
        axes[1, 1].axhline(0, color="gray", linewidth=0.8)
        axes[1, 1].set_title("Estimated entropy decay rate")
        filename = "fig5_3"
    for ax in axes.flat:
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    replicates = sorted({len(group) for group in groups.values()})
    note = "single n; convergence requires an n sweep" if len(ns) == 1 else f"{len(ns)} particle counts"
    fig.suptitle(f"IHW25 {filename.replace('_', '.').replace('fig', 'Figure ')} diagnostics | "
                 f"d={config['dv']}, B={config['B']:.6g} | {note}\n"
                 f"Means over {','.join(map(str, replicates))} seed(s) per method/n; current repository score methods",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    target = output_dir / f"{filename}.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    with (output_dir / f"{filename}_convergence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "method", "n", "runs", "mean", "std"])
        writer.writeheader()
        writer.writerows(exported)
    manifest = dict(example=example, dv=config["dv"], n_values=ns, methods=methods,
                    convergence_slopes=slopes, figure=str(target),
                    covariance_statistic="raw second moment mean(v v^T), Frobenius norm (not squared)",
                    runs=[dict(path=run["path"], n=run["config"]["n"], method=run["config"]["score_method"],
                               seed=run["config"]["seed"]) for run in runs])
    (output_dir / f"{filename}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target
