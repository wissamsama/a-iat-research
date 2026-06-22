from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from floodcastbench_utils import EXPECTED_TOP_LEVEL, build_manifest_rows, ensure_data_dir, ensure_output_dir, write_csv


def parse_args():
    p = argparse.ArgumentParser(description="Inspect FloodCastBench folder structure.")
    p.add_argument("--data_dir", type=Path, default=Path("data/FloodCastBench"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/dataset_profile"))
    p.add_argument("--max_files_per_group", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args(); data_dir = ensure_data_dir(args.data_dir); output_dir = ensure_output_dir(args.output_dir)
    rows = build_manifest_rows(data_dir)
    top_levels = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    events = sorted({r["event"] for r in rows if r["event"] != "unknown"})
    variables = sorted({r["variable_guess"] for r in rows})
    event_rows = []
    for event in events:
        erows = [r for r in rows if r["event"] == event]
        event_rows.append({"event": event, "num_files": len(erows), "num_rasters": sum(1 for r in erows if r["is_raster"]), "variables": ";".join(sorted({r["variable_guess"] for r in erows})), "resolutions": ";".join(sorted({r["resolution"] for r in erows if r["resolution"] != "unknown"}))})
    variable_rows = []
    for var in variables:
        vrows = [r for r in rows if r["variable_guess"] == var]
        variable_rows.append({"variable_guess": var, "num_files": len(vrows), "num_rasters": sum(1 for r in vrows if r["is_raster"]), "events": ";".join(sorted({r["event"] for r in vrows if r["event"] != "unknown"})), "resolutions": ";".join(sorted({r["resolution"] for r in vrows if r["resolution"] != "unknown"}))})
    write_csv(output_dir / "events_summary.csv", event_rows)
    write_csv(output_dir / "variables_summary.csv", variable_rows)
    md = ["# FloodCastBench Structure Inspection\n", f"Dataset inspected: `{data_dir.as_posix()}`", "\n## Top-level folders"]
    md += [f"- `{x}`" for x in top_levels]
    md.append("\n## Expected folder presence")
    for display, actual in EXPECTED_TOP_LEVEL.items():
        md.append(f"- {display} -> `{actual}`: {actual in top_levels}")
    md.append("\n## Events found")
    md += [f"- `{r['event']}`: files={r['num_files']}, rasters={r['num_rasters']}, variables={r['variables']}, resolutions={r['resolutions']}" for r in event_rows]
    md.append("\n## Subfolder preview")
    for item in sorted(data_dir.rglob("*"), key=lambda p: p.as_posix().lower()):
        rel = item.relative_to(data_dir)
        if item.is_dir() and len(rel.parts) <= 4:
            md.append("  " * (len(rel.parts)-1) + f"- {item.name}/")
    md.append("\n## File extension counts")
    for ext, count in sorted(Counter(r["extension"] for r in rows).items()):
        md.append(f"- `{ext}`: {count}")
    (output_dir / "floodcastbench_structure.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {output_dir / 'floodcastbench_structure.md'}")

if __name__ == "__main__":
    main()
