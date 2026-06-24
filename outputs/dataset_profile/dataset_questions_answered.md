# FloodCastBench Dataset Questions Answered

## What we know from actual files
- Dataset folder exists: `data/FloodCastBench/`.
- Total files: `14868`.
- File extensions: `.tfw`=12, `.tif`=14855, `.txt`=1.
- TIFF/TIFF rasters: `14855`.
- Top-level folders: `High-fidelity flood forecasting`, `Low-fidelity flood forecasting`, `Relevant data`.
- Events found: `Australia flood`, `Mozambique flood`, `Pakistan flood`, `UK flood`.
- Variables inferred from paths: `dem`, `georeference`, `initial_condition`, `lulc`, `rainfall`, `water_depth_forecast`.
- Water-depth forecast rasters are under `Low-fidelity flood forecasting/` and `High-fidelity flood forecasting/`.
- Rainfall rasters are under `Relevant data/Rainfall/`.
- DEM rasters are under `Relevant data/DEM/`.
- LULC rasters are under `Relevant data/Land use and land cover/`.
- Initial condition rasters are under `Relevant data/Initial conditions/`.
- Georeference files are under `Relevant data/Georeferenced files/`.
- No explicit flood mask raster folder was found. Flood masks should be derived by thresholding water-depth rasters.
- Propagation paths can be derived as `future_mask AND NOT current_mask` once a water-depth threshold is chosen.

## Actual folders, events, resolutions and variables
- `Australia flood`: files=6250, rasters=6247, variables=dem;georeference;initial_condition;lulc;rainfall;water_depth_forecast, resolutions=30m;60m
- `Mozambique flood`: files=2024, rasters=2021, variables=dem;georeference;initial_condition;lulc;rainfall;water_depth_forecast, resolutions=480m
- `Pakistan flood`: files=4711, rasters=4708, variables=dem;georeference;initial_condition;lulc;rainfall;water_depth_forecast, resolutions=480m
- `UK flood`: files=1882, rasters=1879, variables=dem;georeference;initial_condition;lulc;rainfall;water_depth_forecast, resolutions=30m;60m
- `dem`: files=8, rasters=4, events=Australia flood;Mozambique flood;Pakistan flood;UK flood, resolutions=
- `georeference`: files=5, rasters=0, events=Australia flood;Mozambique flood;Pakistan flood;UK flood, resolutions=
- `initial_condition`: files=6, rasters=6, events=Australia flood;Mozambique flood;Pakistan flood;UK flood, resolutions=30m;480m;60m
- `lulc`: files=8, rasters=4, events=Australia flood;Mozambique flood;Pakistan flood;UK flood, resolutions=
- `rainfall`: files=1587, rasters=1587, events=Australia flood;Mozambique flood;Pakistan flood;UK flood, resolutions=
- `water_depth_forecast`: files=13254, rasters=13254, events=Australia flood;Mozambique flood;Pakistan flood;UK flood, resolutions=30m;480m;60m

## Actual raster dimensions and metadata sampled
- `initial_condition` `Australia flood` `30m`: 1073x1073, bands=1, dtype=float32, crs=missing, nodata=none
- `water_depth_forecast` `Australia flood` `30m`: 1073x1073, bands=1, dtype=float32, crs=missing, nodata=none
- `initial_condition` `Australia flood` `60m`: 536x536, bands=1, dtype=float32, crs=missing, nodata=none
- `water_depth_forecast` `Australia flood` `60m`: 536x536, bands=1, dtype=float32, crs=missing, nodata=none
- `dem` `Australia flood` `unknown`: 1073x1073, bands=1, dtype=float32, crs=EPSG:32756, nodata=-3.4028234663852886e+38
- `lulc` `Australia flood` `unknown`: 1073x1073, bands=1, dtype=int32, crs=EPSG:32756, nodata=0.0
- `rainfall` `Australia flood` `unknown`: 1073x1073, bands=1, dtype=float32, crs=EPSG:32756, nodata=-3.4028230607370965e+38
- `initial_condition` `Mozambique flood` `480m`: 151x138, bands=1, dtype=float32, crs=missing, nodata=none
- `water_depth_forecast` `Mozambique flood` `480m`: 151x138, bands=1, dtype=float32, crs=missing, nodata=none
- `dem` `Mozambique flood` `unknown`: 2624x2321, bands=1, dtype=float32, crs=EPSG:32736, nodata=-3.4028234663852886e+38
- `lulc` `Mozambique flood` `unknown`: 2624x2321, bands=1, dtype=uint8, crs=EPSG:32736, nodata=0.0
- `rainfall` `Mozambique flood` `unknown`: 2624x2321, bands=1, dtype=float32, crs=EPSG:32736, nodata=-3.4028230607370965e+38
- `initial_condition` `Pakistan flood` `480m`: 810x441, bands=1, dtype=float32, crs=missing, nodata=none
- `water_depth_forecast` `Pakistan flood` `480m`: 810x441, bands=1, dtype=float32, crs=missing, nodata=none
- `dem` `Pakistan flood` `unknown`: 13035x7298, bands=1, dtype=float32, crs=EPSG:32642, nodata=-3.4028234663852886e+38
- `lulc` `Pakistan flood` `unknown`: 13035x7298, bands=1, dtype=uint8, crs=EPSG:32642, nodata=15.0
- `rainfall` `Pakistan flood` `unknown`: 13035x7298, bands=1, dtype=float32, crs=EPSG:32642, nodata=-3.4028230607370965e+38
- `initial_condition` `UK flood` `30m`: 170x275, bands=1, dtype=float32, crs=missing, nodata=none
- `water_depth_forecast` `UK flood` `30m`: 170x275, bands=1, dtype=float32, crs=missing, nodata=none
- `initial_condition` `UK flood` `60m`: 85x137, bands=1, dtype=float32, crs=missing, nodata=none
- `water_depth_forecast` `UK flood` `60m`: 85x137, bands=1, dtype=float32, crs=missing, nodata=none
- `dem` `UK flood` `unknown`: 180x285, bands=1, dtype=float32, crs=EPSG:32630, nodata=-3.4028234663852886e+38
- `lulc` `UK flood` `unknown`: 180x285, bands=1, dtype=uint8, crs=EPSG:32630, nodata=15.0
- `rainfall` `UK flood` `unknown`: 180x285, bands=1, dtype=float32, crs=EPSG:32630, nodata=-3.4028230607370965e+38

## Temporal organization
- `water_depth_forecast` `Australia flood` `30m`: steps=2881, min=0, max=864000, median_step=300, missing_steps=0.
- `water_depth_forecast` `Australia flood` `60m`: steps=2881, min=0, max=864000, median_step=300, missing_steps=0.
- `rainfall` `Australia flood` `unknown`: steps=481, min=, max=, median_step=, missing_steps=0.
- `water_depth_forecast` `Mozambique flood` `480m`: steps=1729, min=0, max=518400, median_step=300, missing_steps=0.
- `rainfall` `Mozambique flood` `unknown`: steps=289, min=, max=, median_step=, missing_steps=0.
- `water_depth_forecast` `Pakistan flood` `480m`: steps=4033, min=0, max=1209600, median_step=300, missing_steps=0.
- `rainfall` `Pakistan flood` `unknown`: steps=672, min=, max=, median_step=, missing_steps=0.
- `water_depth_forecast` `UK flood` `30m`: steps=865, min=0, max=259200, median_step=300, missing_steps=0.
- `water_depth_forecast` `UK flood` `60m`: steps=865, min=0, max=259200, median_step=300, missing_steps=0.
- `rainfall` `UK flood` `unknown`: steps=145, min=, max=, median_step=, missing_steps=0.
- Integer-named water-depth files use 300-second spacing. Therefore T+20 = 100 minutes, T+6h = 72 steps, T+12h = 144 steps.
- Rainfall files use timestamp-like filenames; numeric step spacing is not encoded the same way in the filename field.
- DEM and LULC are static files, not repeated over time.
- Initial conditions are separate static/initial rasters, not part of the forecast time sequence.

## Benchmark samples vs raw time steps
- The local files contain raw temporal raster frames plus relevant static/dynamic rasters.
- No local train/validation/test split files were found in the manifest.
- No local benchmark-window definition files were found in the manifest.
- Reported benchmark sample counts from the paper do not directly match raw TIFF frame counts because the local folder contains raw frames, not only precomputed benchmark windows.

## Horizons and supervised samples
- `Australia flood` `30m` input_window=5 horizon=20: `2857` possible samples.
- `Australia flood` `30m` input_window=5 horizon=72: `2805` possible samples.
- `Australia flood` `30m` input_window=5 horizon=144: `2733` possible samples.
- `Australia flood` `60m` input_window=5 horizon=20: `2857` possible samples.
- `Australia flood` `60m` input_window=5 horizon=72: `2805` possible samples.
- `Australia flood` `60m` input_window=5 horizon=144: `2733` possible samples.
- `Mozambique flood` `480m` input_window=5 horizon=20: `1705` possible samples.
- `Mozambique flood` `480m` input_window=5 horizon=72: `1653` possible samples.
- `Mozambique flood` `480m` input_window=5 horizon=144: `1581` possible samples.
- `Pakistan flood` `480m` input_window=5 horizon=20: `4009` possible samples.
- `Pakistan flood` `480m` input_window=5 horizon=72: `3957` possible samples.
- `Pakistan flood` `480m` input_window=5 horizon=144: `3885` possible samples.
- `UK flood` `30m` input_window=5 horizon=20: `841` possible samples.
- `UK flood` `30m` input_window=5 horizon=72: `789` possible samples.
- `UK flood` `30m` input_window=5 horizon=144: `717` possible samples.
- `UK flood` `60m` input_window=5 horizon=20: `841` possible samples.
- `UK flood` `60m` input_window=5 horizon=72: `789` possible samples.
- `UK flood` `60m` input_window=5 horizon=144: `717` possible samples.

## Supervised learning formulation supported by local files
- Input tensor can use past water-depth rasters plus rainfall sequence plus static DEM/LULC plus initial condition, if spatial alignment is handled per event/resolution.
- Target tensor can be a future water-depth raster at horizon h.
- Dynamic variables found: water-depth forecast rasters and rainfall rasters.
- Static variables found: DEM, LULC, georeference sidecars, initial condition rasters.
- Primary target/label for forecasting should be future water-depth raster.
- Binary masks can be derived with configurable thresholds such as 0.001 m and 0.01 m.
- Propagation path can be `future_mask AND NOT current_mask`.

## Data leakage risks
- Random windows from the same event can leak adjacent frames across train/val/test.
- A temporal split inside one event should include a gap at least as large as input_window + horizon around boundaries.
- Event-level split is safest for cross-region evaluation, but events differ by fidelity/resolution and shape.
- Static DEM/LULC can leak location when the same region appears in train and test.
- Georeferenced metadata should initially be metadata only, not a model input.

## Recommended first training setup
- Start with `High-fidelity flood forecasting/30m/Australia` because it is high resolution and has 2881 regularly spaced frames.
- First variables: past water-depth only.
- First target: future water-depth raster.
- First horizon: 20 steps (100 minutes).
- First input window: 5 frames.
- Expected single-frame tensor shape for Australia 30m: `[1, 1073, 1073]`; with input_window=5: `[5, 1, 1073, 1073]` before batching.
- Use lazy raster loading and optional patching/cropping; full 1073x1073 batches may be memory-heavy.
- Normalize water depth and rainfall separately. Treat LULC as categorical, not continuous elevation.

## What we still do not know
- Physical units are not explicitly encoded in TIFF metadata; water depth in meters is likely but requires official documentation confirmation.
- Official train/validation/test splits are not present as local files.
- Official benchmark-window construction is not present as local files.
- Forecast TIFFs sampled have missing CRS metadata/identity transform, while relevant rainfall files have CRS. Exact alignment must be checked before multi-variable fusion.
- Velocity, discharge, boundary-condition files, and explicit flood masks were not identified in local paths.

## What must be verified manually
- Confirm units for water depth and rainfall.
- Confirm official benchmark protocol and splits.
- Confirm whether forecast rasters should be georeferenced using sidecar files in `Relevant data/Georeferenced files/`.
- Confirm intended cross-region train/validation/test setup before claiming benchmark comparability.