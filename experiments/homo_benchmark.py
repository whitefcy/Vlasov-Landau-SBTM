"""Run homogeneous particle-count/seed sweeps and assemble IHW25 figures.

python experiments/homo_benchmark.py run --example anisotropic --output_dir data/my_sweep
python experiments/homo_benchmark.py plot --input_dirs data/my_sweep --output_dir data/my_plots

Unrecognized options after 'run' are validated by and forwarded to each
experiment, e.g. --dv 10, --dt 0.005, or --sbtm_num_epochs 2000.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", allow_abbrev=False)
    run.add_argument("--example", choices=["bkw", "anisotropic"], required=True)
    run.add_argument("--n_values", nargs="+", type=int, default=[100, 200, 400, 800, 1600, 3200, 6400, 12800])
    run.add_argument("--score_methods", nargs="+", choices=["sbtm", "blob", "exact"], default=["sbtm", "blob"])
    run.add_argument("--seeds", nargs="+", type=int, default=[42])
    run.add_argument("--output_dir", type=Path, default=None)
    plot = sub.add_parser("plot", allow_abbrev=False)
    plot.add_argument("--input_dirs", type=Path, nargs="+", required=True)
    plot.add_argument("--output_dir", type=Path, required=True)
    args, extra = parser.parse_known_args(argv)
    if args.action == "plot":
        if extra:
            parser.error(f"Unrecognized plot options: {extra}")
        os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / ".matplotlib"))
        from src.homogeneous_plots import load_runs, plot_benchmarks
        result = plot_benchmarks(load_runs(args.input_dirs), args.output_dir)
        print(f"Figure: {result}", flush=True)
        return result

    for name in ["n_values", "score_methods", "seeds"]:
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate entries in {name}")
    if min(args.n_values) < 2:
        parser.error("All particle counts must be >= 2")
    if any(token.split("=", 1)[0] in {"--n", "--score_method", "--seed", "--wandb_run_name"} for token in extra):
        parser.error("Use n_values, score_methods, and seeds to select sweep runs")
    if args.example == "bkw":
        from experiments.homo_BKW import parse_args as validate
        script = "homo_BKW.py"
    else:
        from experiments.homo_anisotropic import parse_args as validate
        script = "homo_anisotropic.py"
    for method in args.score_methods:
        validate([*extra, "--n", str(args.n_values[0]), "--score_method", method])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = "homo_BKW" if args.example == "bkw" else "homo_anisotropic"
    outdir = args.output_dir or Path("data") / folder / f"sweep_{stamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(outdir.resolve() / ".matplotlib"))
    specification = dict(example=args.example, n_values=args.n_values, score_methods=args.score_methods,
                         seeds=args.seeds, experiment_arguments=extra)
    (outdir / "sweep.json").write_text(json.dumps(specification, indent=2) + "\n")
    paths = []
    for n in args.n_values:
        for method in args.score_methods:
            for seed in args.seeds:
                name = f"{args.example}_n{n}_{method}_seed{seed}"
                root = outdir / name
                command = [sys.executable, str(Path(__file__).with_name(script)), *extra,
                           "--n", str(n), "--score_method", method, "--seed", str(seed),
                           "--output_dir", str(root), "--wandb_run_name", name]
                print(f"Running {name}", flush=True)
                # Separate processes release JAX compilation/device memory between runs.
                subprocess.run(command, check=True)
                paths.append(root)
    from src.homogeneous_plots import load_runs, plot_benchmarks
    result = plot_benchmarks(load_runs(paths), outdir)
    print(f"Sweep complete. Combined figure: {result}", flush=True)
    return result


if __name__ == "__main__":
    main()
