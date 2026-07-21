#!/usr/bin/env python3
"""Produce a machine-local copy of a config with the `paths:` block
rewritten to this DGX Spark's real absolute workspace path.

Needed because tools/evaluate_floodcastbench_diff_sparse_v2.py (unlike the
trainer) has no --dataset-root/--experiment-root/--checkpoint-root/--log-
root override flags -- it reads paths.* straight from the config, which on
every committed config hardcodes /home/wissam/utem-workspace/... (correct
on P7/Dell, not on this standalone machine -- no /home/wissam symlink, and
deliberately not creating one outside the sandbox, see git history).

Output configs are written under local_spark_configs/ (gitignored-by-
convention -- do not commit these, they contain a machine-specific
absolute path) and are otherwise byte-identical to the source config.

Usage:
    python3 scripts/make_spark_local_config.py configs/some_config.yaml
    -> prints the path to the patched local copy
"""
import sys
from pathlib import Path

import yaml

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: make_spark_local_config.py <config.yaml>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    repo_dir = Path(__file__).resolve().parents[1]
    workspace = repo_dir.parents[1]  # code/a-iat-research -> code -> utem-workspace

    config = yaml.safe_load(src.read_text())
    config.setdefault("paths", {})
    config["paths"]["dataset_root"] = str(workspace / "data" / "FloodCastBench")
    config["paths"]["experiment_root"] = str(workspace / "experiments" / "FloodCastBench")
    config["paths"]["checkpoint_root"] = str(workspace / "checkpoints" / "FloodCastBench")
    config["paths"]["log_root"] = str(workspace / "logs" / "FloodCastBench")

    out_dir = repo_dir / "local_spark_configs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / src.name
    out_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
