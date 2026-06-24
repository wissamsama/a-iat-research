# FloodCastBench Structure Inspection

Dataset inspected: `data/FloodCastBench`

## Top-level folders
- `High-fidelity flood forecasting`
- `Low-fidelity flood forecasting`
- `Relevant data`

## Expected folder presence
- Low-Fidelity Flood Forecasting -> `Low-fidelity flood forecasting`: True
- High-Fidelity Flood Forecasting -> `High-fidelity flood forecasting`: True
- Relevant Data -> `Relevant data`: True

## Events found
- `Australia flood`: files=6250, rasters=6247, variables=dem;georeference;initial_condition;lulc;rainfall;water_depth_forecast, resolutions=30m;60m
- `Mozambique flood`: files=2024, rasters=2021, variables=dem;georeference;initial_condition;lulc;rainfall;water_depth_forecast, resolutions=480m
- `Pakistan flood`: files=4711, rasters=4708, variables=dem;georeference;initial_condition;lulc;rainfall;water_depth_forecast, resolutions=480m
- `UK flood`: files=1882, rasters=1879, variables=dem;georeference;initial_condition;lulc;rainfall;water_depth_forecast, resolutions=30m;60m

## Subfolder preview
- High-fidelity flood forecasting/
  - 30m/
    - Australia/
    - UK/
  - 60m/
    - Australia/
    - UK/
- Low-fidelity flood forecasting/
  - 480m/
    - Mozambique/
    - Pakistan/
- Relevant data/
  - DEM/
  - Georeferenced files/
  - Initial conditions/
    - High-fidelity flood forecasting/
    - Low-fidelity flood forecasting/
  - Land use and land cover/
  - Rainfall/
    - Australia flood/
    - Mozambique flood/
    - Pakistan flood/
    - UK flood/

## File extension counts
- `.tfw`: 12
- `.tif`: 14855
- `.txt`: 1