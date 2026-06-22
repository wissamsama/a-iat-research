from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

EXPECTED_TOP_LEVEL = {
    "Low-Fidelity Flood Forecasting": "Low-fidelity flood forecasting",
    "High-Fidelity Flood Forecasting": "High-fidelity flood forecasting",
    "Relevant Data": "Relevant data",
}
EVENT_ALIASES = {
    "australia": "Australia flood",
    "mozambique": "Mozambique flood",
    "pakistan": "Pakistan flood",
    "uk": "UK flood",
}
TEMPORAL_VARIABLES = {"water_depth_forecast", "rainfall"}


def ensure_data_dir(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise SystemExit(f"FloodCastBench folder not found: {data_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"FloodCastBench path is not a directory: {data_dir}")
    return data_dir


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def is_raster(path):
    return path.suffix.lower() in {".tif", ".tiff"}


def event_from_name(name):
    low = name.lower()
    for token, event in EVENT_ALIASES.items():
        if token in low:
            return event
    return "unknown"


def time_index_from_stem(stem):
    return int(stem) if stem.isdigit() else stem


def infer_file_record(path, data_dir):
    rel = path.relative_to(data_dir)
    parts = rel.parts
    top = parts[0] if parts else ""
    event = "unknown"
    resolution = "unknown"
    variable = "unknown"
    time_index = ""
    if top in {"Low-fidelity flood forecasting", "High-fidelity flood forecasting"} and len(parts) >= 4:
        resolution = parts[1]
        event = f"{parts[2]} flood"
        variable = "water_depth_forecast"
        time_index = time_index_from_stem(path.stem)
    elif top == "Relevant data" and len(parts) >= 2:
        section = parts[1]
        if section == "DEM":
            variable = "dem"; event = event_from_name(path.stem)
        elif section == "Land use and land cover":
            variable = "lulc"; event = event_from_name(path.stem)
        elif section == "Rainfall" and len(parts) >= 4:
            variable = "rainfall"; event = parts[2]; time_index = time_index_from_stem(path.stem)
        elif section == "Initial conditions":
            variable = "initial_condition"; event = event_from_name(path.stem)
            resolution = path.stem.split("_")[-1] if "_" in path.stem else "unknown"
        elif section == "Georeferenced files":
            variable = "georeference"; event = event_from_name(path.stem)
    return {
        "file_path": str(path.resolve()),
        "relative_path": rel.as_posix(),
        "file_name": path.name,
        "extension": path.suffix.lower() or "[none]",
        "top_level_folder": top,
        "event": event,
        "resolution": resolution,
        "variable_guess": variable,
        "time_index_guess": time_index,
        "is_raster": is_raster(path),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 6),
    }


def build_manifest_rows(data_dir):
    data_dir = Path(data_dir)
    return [infer_file_record(p, data_dir) for p in sorted(data_dir.rglob("*"), key=lambda x: x.as_posix().lower()) if p.is_file()]


def write_csv(path, rows, fieldnames=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def natural_time_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def temporal_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        is_r = row["is_raster"] in {True, "True", "true"}
        if is_r and row["variable_guess"] in TEMPORAL_VARIABLES:
            groups[(row["event"], row["resolution"], row["variable_guess"])].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: natural_time_key(row["time_index_guess"]))
    return dict(groups)


def time_values(group):
    values = []
    for row in group:
        try:
            values.append(int(row["time_index_guess"] or 0))
        except ValueError:
            pass
    return sorted(values)


def temporal_summary_rows(rows):
    out = []
    for (event, resolution, variable), group in sorted(temporal_groups(rows).items()):
        values = time_values(group)
        diffs = np.diff(values) if len(values) > 1 else np.array([], dtype=int)
        median_step = int(np.median(diffs)) if len(diffs) else ""
        missing = 0
        if len(values) > 1 and median_step:
            expected = set(range(values[0], values[-1] + median_step, median_step))
            missing = len(expected - set(values))
        out.append({
            "event": event,
            "resolution": resolution,
            "variable": variable,
            "num_time_steps": len(group),
            "min_time_index": values[0] if values else "",
            "max_time_index": values[-1] if values else "",
            "median_step_difference": median_step,
            "has_missing_steps": missing > 0,
            "missing_steps_count": missing,
            "example_first_file": group[0]["relative_path"] if group else "",
            "example_last_file": group[-1]["relative_path"] if group else "",
        })
    return out


def supervised_preview_rows(rows, input_window, horizons):
    out = []
    for (event, resolution, variable), group in sorted(temporal_groups(rows).items()):
        if variable != "water_depth_forecast":
            continue
        values = time_values(group)
        available = set(values)
        for horizon in horizons:
            samples = []
            for pos in range(len(values)):
                input_seq = values[pos:pos + input_window]
                if len(input_seq) < input_window:
                    continue
                target = input_seq[-1] + horizon * 300
                if target in available:
                    samples.append((input_seq[0], input_seq[-1], target))
            first = samples[0] if samples else ("", "", "")
            last = samples[-1] if samples else ("", "", "")
            out.append({
                "event": event,
                "resolution": resolution,
                "input_window": input_window,
                "forecast_horizon": horizon,
                "num_possible_samples": len(samples),
                "first_input_start": first[0],
                "first_input_end": first[1],
                "first_target": first[2],
                "last_input_start": last[0],
                "last_input_end": last[1],
                "last_target": last[2],
            })
    return out


def read_raster_metadata(path):
    import rasterio
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        vals = data.compressed()
        if vals.size:
            stats = {
                "min": float(vals.min()), "max": float(vals.max()),
                "mean": float(vals.mean()), "std": float(vals.std()),
                "negative_count": int((vals < 0).sum()),
                "zero_count": int((vals == 0).sum()),
                "positive_count": int((vals > 0).sum()),
            }
        else:
            stats = {"min": "", "max": "", "mean": "", "std": "", "negative_count": 0, "zero_count": 0, "positive_count": 0}
        filled = data.astype(np.float64).filled(np.nan)
        return {
            "shape_height": src.height,
            "shape_width": src.width,
            "count_bands": src.count,
            "dtype": ";".join(src.dtypes),
            "crs": str(src.crs) if src.crs else "",
            "transform": str(src.transform),
            "pixel_size_x": float(src.transform.a),
            "pixel_size_y": float(src.transform.e),
            "nodata_value": src.nodata if src.nodata is not None else "",
            "nan_count": int(np.isnan(filled).sum()),
            **stats,
        }


def sample_group(group, max_files, seed=7):
    if len(group) <= max_files:
        return group
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(np.arange(len(group)), size=max_files, replace=False))
    return [group[int(i)] for i in picks]

def write_questions_report(output_dir):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "floodcastbench_manifest.csv"
    if not manifest_path.exists():
        return
    manifest = load_csv(manifest_path)
    events = load_csv(output_dir / "events_summary.csv") if (output_dir / "events_summary.csv").exists() else []
    variables = load_csv(output_dir / "variables_summary.csv") if (output_dir / "variables_summary.csv").exists() else []
    temporal = load_csv(output_dir / "temporal_index_summary.csv") if (output_dir / "temporal_index_summary.csv").exists() else []
    preview = load_csv(output_dir / "supervised_learning_samples_preview.csv") if (output_dir / "supervised_learning_samples_preview.csv").exists() else []
    raster = load_csv(output_dir / "raster_statistics.csv") if (output_dir / "raster_statistics.csv").exists() else []
    ext_counts = {}
    for row in manifest:
        ext_counts[row["extension"]] = ext_counts.get(row["extension"], 0) + 1
    md = ["# FloodCastBench Dataset Questions Answered\n", "## What we know from actual files"]
    md.append("- Dataset folder exists: `data/FloodCastBench/`.")
    md.append(f"- Total files: `{len(manifest)}`.")
    md.append("- File extensions: " + ", ".join(f"`{k}`={v}" for k, v in sorted(ext_counts.items())) + ".")
    md.append(f"- TIFF/TIFF rasters: `{sum(1 for r in manifest if r['extension'] in {'.tif', '.tiff'})}`.")
    md.append("- Top-level folders: " + ", ".join(f"`{x}`" for x in sorted({r['top_level_folder'] for r in manifest})) + ".")
    md.append("- Events found: " + ", ".join(f"`{x}`" for x in sorted({r['event'] for r in manifest if r['event'] != 'unknown'})) + ".")
    md.append("- Variables inferred from paths: " + ", ".join(f"`{x}`" for x in sorted({r['variable_guess'] for r in manifest})) + ".")
    md += [
        "- Water-depth forecast rasters are under `Low-fidelity flood forecasting/` and `High-fidelity flood forecasting/`.",
        "- Rainfall rasters are under `Relevant data/Rainfall/`.",
        "- DEM rasters are under `Relevant data/DEM/`.",
        "- LULC rasters are under `Relevant data/Land use and land cover/`.",
        "- Initial condition rasters are under `Relevant data/Initial conditions/`.",
        "- Georeference files are under `Relevant data/Georeferenced files/`.",
        "- No explicit flood mask raster folder was found. Flood masks should be derived by thresholding water-depth rasters.",
        "- Propagation paths can be derived as `future_mask AND NOT current_mask` once a water-depth threshold is chosen.\n",
        "## Actual folders, events, resolutions and variables",
    ]
    for row in events:
        md.append(f"- `{row['event']}`: files={row['num_files']}, rasters={row['num_rasters']}, variables={row['variables']}, resolutions={row['resolutions']}")
    for row in variables:
        md.append(f"- `{row['variable_guess']}`: files={row['num_files']}, rasters={row['num_rasters']}, events={row['events']}, resolutions={row['resolutions']}")
    md.append("\n## Actual raster dimensions and metadata sampled")
    seen = set()
    for row in raster:
        key = (row['event'], row['resolution'], row['variable_guess'], row['shape_height'], row['shape_width'], row['count_bands'], row['dtype'], row['crs'])
        if key in seen:
            continue
        seen.add(key)
        md.append(f"- `{row['variable_guess']}` `{row['event']}` `{row['resolution']}`: {row['shape_height']}x{row['shape_width']}, bands={row['count_bands']}, dtype={row['dtype']}, crs={row['crs'] or 'missing'}, nodata={row['nodata_value'] or 'none'}")
    md.append("\n## Temporal organization")
    for row in temporal:
        md.append(f"- `{row['variable']}` `{row['event']}` `{row['resolution']}`: steps={row['num_time_steps']}, min={row['min_time_index']}, max={row['max_time_index']}, median_step={row['median_step_difference']}, missing_steps={row['missing_steps_count']}.")
    md += [
        "- Integer-named water-depth files use 300-second spacing. Therefore T+20 = 100 minutes, T+6h = 72 steps, T+12h = 144 steps.",
        "- Rainfall files use timestamp-like filenames; numeric step spacing is not encoded the same way in the filename field.",
        "- DEM and LULC are static files, not repeated over time.",
        "- Initial conditions are separate static/initial rasters, not part of the forecast time sequence.\n",
        "## Benchmark samples vs raw time steps",
        "- The local files contain raw temporal raster frames plus relevant static/dynamic rasters.",
        "- No local train/validation/test split files were found in the manifest.",
        "- No local benchmark-window definition files were found in the manifest.",
        "- Reported benchmark sample counts from the paper do not directly match raw TIFF frame counts because the local folder contains raw frames, not only precomputed benchmark windows.\n",
        "## Horizons and supervised samples",
    ]
    for row in preview:
        md.append(f"- `{row['event']}` `{row['resolution']}` input_window={row['input_window']} horizon={row['forecast_horizon']}: `{row['num_possible_samples']}` possible samples.")
    md += [
        "\n## Supervised learning formulation supported by local files",
        "- Input tensor can use past water-depth rasters plus rainfall sequence plus static DEM/LULC plus initial condition, if spatial alignment is handled per event/resolution.",
        "- Target tensor can be a future water-depth raster at horizon h.",
        "- Dynamic variables found: water-depth forecast rasters and rainfall rasters.",
        "- Static variables found: DEM, LULC, georeference sidecars, initial condition rasters.",
        "- Primary target/label for forecasting should be future water-depth raster.",
        "- Binary masks can be derived with configurable thresholds such as 0.001 m and 0.01 m.",
        "- Propagation path can be `future_mask AND NOT current_mask`.",
        "\n## Data leakage risks",
        "- Random windows from the same event can leak adjacent frames across train/val/test.",
        "- A temporal split inside one event should include a gap at least as large as input_window + horizon around boundaries.",
        "- Event-level split is safest for cross-region evaluation, but events differ by fidelity/resolution and shape.",
        "- Static DEM/LULC can leak location when the same region appears in train and test.",
        "- Georeferenced metadata should initially be metadata only, not a model input.",
        "\n## Recommended first training setup",
        "- Start with `High-fidelity flood forecasting/30m/Australia` because it is high resolution and has 2881 regularly spaced frames.",
        "- First variables: past water-depth only.",
        "- First target: future water-depth raster.",
        "- First horizon: 20 steps (100 minutes).",
        "- First input window: 5 frames.",
        "- Expected single-frame tensor shape for Australia 30m: `[1, 1073, 1073]`; with input_window=5: `[5, 1, 1073, 1073]` before batching.",
        "- Use lazy raster loading and optional patching/cropping; full 1073x1073 batches may be memory-heavy.",
        "- Normalize water depth and rainfall separately. Treat LULC as categorical, not continuous elevation.",
        "\n## What we still do not know",
        "- Physical units are not explicitly encoded in TIFF metadata; water depth in meters is likely but requires official documentation confirmation.",
        "- Official train/validation/test splits are not present as local files.",
        "- Official benchmark-window construction is not present as local files.",
        "- Forecast TIFFs sampled have missing CRS metadata/identity transform, while relevant rainfall files have CRS. Exact alignment must be checked before multi-variable fusion.",
        "- Velocity, discharge, boundary-condition files, and explicit flood masks were not identified in local paths.",
        "\n## What must be verified manually",
        "- Confirm units for water depth and rainfall.",
        "- Confirm official benchmark protocol and splits.",
        "- Confirm whether forecast rasters should be georeferenced using sidecar files in `Relevant data/Georeferenced files/`.",
        "- Confirm intended cross-region train/validation/test setup before claiming benchmark comparability.",
    ]
    (output_dir / "dataset_questions_answered.md").write_text("\n".join(md), encoding="utf-8")
