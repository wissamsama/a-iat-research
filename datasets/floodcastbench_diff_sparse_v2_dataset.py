from __future__ import annotations

"""DIFF-SPARSE V2 dataset: a thin subclass of the (frozen) V1 dataset.

Adds, without touching any V1 file:
  1. Dihedral data augmentation (8 flips/rotations) for training samples,
     applied consistently to every spatial tensor. Physically consistent:
     gravity enters only through the DEM channel, which is transformed
     together with water/rain/masks. Relevant because the benchmark is a
     single flood event over a single region (2320 train frames) -- spatial
     augmentation is the main regularization available.
  2. Optional randomized training sparsity: masking.missing_rate_range
     samples a missing rate uniformly per training sample, producing one
     model robust across sparsity levels (the paper's own "different sensor
     configurations without retraining" claim, extended to levels). Off by
     default; evaluation always uses the fixed masking.missing_rate.
  3. STRUCTURED evaluation masks (paper master plan WP7): the benchmark's
     i.i.d. random pixel masks are an unrealistic sensor-network model; real
     deployments are gauges along the drainage network or spatially clustered
     coverage. masking.eval_mask_structure selects the eval-bank generator:
       - 'random'  (default): V1's i.i.d. masks, unchanged.
       - 'gauge'   : sensors sampled with probability proportional to the
                     train-split water-occupancy map (fraction of train
                     frames wet at gamma) -- a proxy for river gauges.
       - 'cluster' : sensors in compact spatial blobs around random centers
                     (uneven regional coverage).
     Training masks stay i.i.d. random: structured sparsity is an EVALUATION
     protocol testing generalization to realistic sensor layouts without
     retraining. Same sensor budget as the corresponding missing_rate.
  4. Optional per-pixel Manning-roughness channel derived from the
     benchmark's LULC raster (paper master plan WP5 / V2.1). The official
     FloodCastBench simulator itself takes per-pixel Manning computed from
     LULC (verified in its Data_Generation_Code); no published baseline uses
     it. dataset.include_manning: true adds sample["manning"] [1, ph, pw],
     nearest-resized (categorical source) and standardized by its own
     map statistics. The code->n lookup is configurable
     (dataset.manning_lookup); defaults are standard Chow/HEC-RAS-style
     values for the (UNVERIFIED, see master plan WP5 prerequisites)
     ESRI 10-class hypothesis -- verify before any paper claim.
"""

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from datasets.floodcastbench_diff_sparse_v1_dataset import (
    FloodCastBenchDiffSparseV1Dataset,
    _read_raster,
    split_frame_ranges,
    window_starts_for_split,
)


_SPATIAL_KEYS = ("context_water_masked", "context_water_true", "sensor_mask", "dem", "rainfall", "target")


class _StridedFrameView:
    """Read-only proxy making V1's frozen ``self.frames[start:start+window_length]``
    slicing (used verbatim inside its unmodified ``__getitem__``) transparently
    return frames spaced ``stride`` apart in the underlying list instead of
    consecutive ones. WP12 dose-response (paper master plan): the crossed
    {Delta t x target representation} design needs one-step training windows
    at Delta t > the native 300s cadence, obtained by subsampling the
    existing frame sequence -- exactly the convention used by
    tools/build_mechanism_dose_response.py's Phase 1 ratio measurement, kept
    consistent here. V1 itself is frozen by project convention (see this
    module's docstring); this proxy lets V2 reuse V1's __getitem__ body
    unmodified rather than duplicating it with one line changed, which would
    be a bigger correctness risk for identical effect.

    V1's __getitem__ always requests exactly one contiguous, step=1 slice of
    length ``window_length`` per call (`frames[start : start + window_length]`)
    -- this proxy supports exactly that pattern and nothing else, deliberately,
    so any future change to V1's indexing contract fails loudly here instead
    of silently returning wrong frames.
    """

    def __init__(self, frames: list, stride: int) -> None:
        self._frames = frames
        self._stride = int(stride)

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError(
                "_StridedFrameView only supports the contiguous step=1 slice "
                "pattern V1's __getitem__ uses (frames[start:start+window_length])"
            )
        start = key.start or 0
        stop = key.stop
        if stop is None:
            raise TypeError("_StridedFrameView requires an explicit slice stop")
        window_length = stop - start
        return self._frames[start : start + window_length * self._stride : self._stride]

    def __len__(self) -> int:
        return len(self._frames)

# Manning's n per LULC code (ESRI/Impact-Observatory 10-class scheme).
# WP5 verification (coordination/reports/0003_lulc_scheme_verification_report.md):
# our observed codes {1,2,4,5,7,8,9,10,11,+15=nodata} match this scheme's
# distinctive "codes 3 and 6 skipped" numbering exactly, and the FloodCastBench
# paper (Nature Scientific Data, s41597-025-04725-2) itself cites a Sentinel-2
# LULC source and per-class Manning's n values matching these class names.
# CAVEAT: the 7 non-fallback values below are transcribed from a web-search
# summary of the paper (Nature full text is paywalled, ResearchGate 403'd,
# arXiv PDF exceeded fetch size) -- not a primary-source-verified table read.
# snow/ice and clouds/nodata are not given by the paper (not real land cover
# in these flood events) and keep their prior HEC-RAS-default placeholders.
DEFAULT_MANNING_LOOKUP: dict[int, float] = {
    1: 0.0350,  # water (paper)
    2: 0.1200,  # trees (paper)
    4: 0.0800,  # flooded vegetation (paper)
    5: 0.0350,  # crops (paper)
    7: 0.3750,  # built area (paper) -- was 0.015, ~25x too low/backwards
    8: 0.0265,  # bare ground (paper)
    9: 0.025,   # snow/ice -- not in paper, unverified HEC-RAS-default placeholder
    10: 0.050,  # clouds (nodata-ish) -- not real land cover, floodplain fallback
    11: 0.0375, # rangeland (paper)
}
# The official simulator's floodplain constant (saint_venant.py); used for
# any unmapped/nodata code.
DEFAULT_MANNING_FALLBACK = 0.05

WET_OCCUPANCY_THRESHOLD_M = 0.001
OCCUPANCY_FRAME_STRIDE = 10  # subsample train frames for the occupancy map


def apply_dihedral(tensor: torch.Tensor, transform_index: int) -> torch.Tensor:
    """Apply one of the 8 dihedral-group transforms to the last two dims."""

    transform_index = int(transform_index) % 8
    rotations = transform_index % 4
    flip = transform_index >= 4
    if flip:
        tensor = torch.flip(tensor, dims=(-1,))
    if rotations:
        tensor = torch.rot90(tensor, k=rotations, dims=(-2, -1))
    return tensor.contiguous()


def generate_gauge_mask(
    height: int,
    width: int,
    missing_rate: float,
    occupancy: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sensors sampled without replacement with probability proportional to
    water occupancy (+ epsilon so dry pixels remain reachable when the budget
    exceeds the wet area). Same sensor budget as the i.i.d. mask."""

    if occupancy.shape != (height, width):
        raise ValueError(f"occupancy must be [{height}, {width}], got {tuple(occupancy.shape)}")
    total = height * width
    sensor_count = int(round((1.0 - float(missing_rate)) * total))
    if sensor_count >= total:
        return torch.ones(1, height, width)
    mask = torch.zeros(total)
    if sensor_count > 0:
        weights = occupancy.flatten().double() + 1e-4
        chosen = torch.multinomial(weights, sensor_count, replacement=False, generator=generator)
        mask[chosen] = 1.0
    return mask.view(1, height, width)


def generate_cluster_mask(
    height: int,
    width: int,
    missing_rate: float,
    generator: torch.Generator | None = None,
    pixels_per_cluster: int = 150,
) -> torch.Tensor:
    """Sensors as compact blobs: random cluster centers, then the
    budget-closest pixels (by distance to the nearest center, tie-broken by
    noise) are selected -- exact same sensor budget as the i.i.d. mask."""

    total = height * width
    sensor_count = int(round((1.0 - float(missing_rate)) * total))
    if sensor_count >= total:
        return torch.ones(1, height, width)
    if sensor_count <= 0:
        return torch.zeros(1, height, width)
    n_clusters = max(1, sensor_count // int(pixels_per_cluster))
    cy = torch.randint(0, height, (n_clusters,), generator=generator).double()
    cx = torch.randint(0, width, (n_clusters,), generator=generator).double()
    ys = torch.arange(height, dtype=torch.float64).view(-1, 1, 1)
    xs = torch.arange(width, dtype=torch.float64).view(1, -1, 1)
    distance_sq = (ys - cy.view(1, 1, -1)) ** 2 + (xs - cx.view(1, 1, -1)) ** 2
    score = distance_sq.min(dim=-1).values
    score = score + 1e-3 * torch.rand(score.shape, generator=generator, dtype=torch.float64)
    threshold = score.flatten().kthvalue(sensor_count).values
    mask = (score <= threshold).float()
    # kthvalue guarantees >= sensor_count selected only if no exact ties above;
    # trim deterministically to the exact budget.
    flat = mask.flatten()
    selected = flat.nonzero(as_tuple=False).flatten()
    if selected.numel() > sensor_count:
        drop = selected[sensor_count:]
        flat[drop] = 0.0
    return flat.view(1, height, width)


class FloodCastBenchDiffSparseV2Dataset(FloodCastBenchDiffSparseV1Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dataset_config = self.config.get("dataset", {})
        masking_config = self.config.get("masking", {})

        # WP12 dose-response (paper master plan): effective time step Delta t
        # via frame subsampling. frame_stride=1 (default) is exactly V1's
        # original behavior, byte-identical -- window_starts is untouched.
        # frame_stride>1 re-derives window_starts because each window now
        # spans window_length*frame_stride raw frames, not window_length.
        self.frame_stride = int(dataset_config.get("frame_stride", 1))
        if self.frame_stride < 1:
            raise ValueError(f"dataset.frame_stride must be >= 1, got {self.frame_stride}")
        if self.frame_stride != 1:
            self.window_starts = window_starts_for_split(
                self.frame_range, self.window_length * self.frame_stride, self.stride
            )
            if not self.window_starts:
                raise ValueError(
                    f"No {self.split} windows fit frame range {self.frame_range} with "
                    f"window_length={self.window_length}, frame_stride={self.frame_stride}, "
                    f"stride={self.stride}"
                )

        # Opt-in in-memory frame cache (default off -- byte-identical output,
        # but a new code path, so existing configs keep their exact original
        # behavior unless they explicitly ask for this). V1's frozen
        # _read_water_patch re-opens and re-decodes the raster file from
        # disk on every __getitem__ call, for every one of window_length
        # frames, every sample, every epoch -- no caching across samples.
        # Diagnosed as the actual bottleneck on large low-fidelity events
        # (e.g. Pakistan, 810x441): profiling showed the DataLoader workers
        # pegged at ~100% CPU while the GPU sat at ~4% utilization, and
        # raising num_workers made things WORSE (already CPU-saturated, not
        # under-parallelized) -- the fix is doing less redundant decode work
        # per sample, not spreading the same work over more workers. A
        # whole event's water-depth frames fit comfortably in memory (e.g.
        # Pakistan: 4033 frames * 810*441*4 bytes ~= 5.8 GB) on any machine
        # with tens of GB of RAM, let alone this project's typical
        # workstations. Deliberately unbounded (no LRU eviction) -- fine at
        # this dataset scale; would need bounding before reuse on a much
        # larger event.
        self.cache_frames_in_memory = bool(dataset_config.get("cache_frames_in_memory", False))
        self._water_frame_cache: dict[Path, np.ndarray] = {}

        self.augmentation = bool(dataset_config.get("augmentation", False)) and self.patch_mode == "random"
        rate_range = masking_config.get("missing_rate_range")
        if rate_range is not None and self.patch_mode == "random":
            low, high = float(rate_range[0]), float(rate_range[1])
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError(f"missing_rate_range must satisfy 0 <= low <= high <= 1, got {rate_range}")
            self.missing_rate_range: tuple[float, float] | None = (low, high)
        else:
            self.missing_rate_range = None

        self.eval_mask_structure = str(masking_config.get("eval_mask_structure", "random")).lower()
        if self.eval_mask_structure not in {"random", "gauge", "cluster"}:
            raise ValueError(
                f"masking.eval_mask_structure must be random|gauge|cluster, got {self.eval_mask_structure!r}"
            )
        self._occupancy_map: torch.Tensor | None = None

        self.include_manning = bool(dataset_config.get("include_manning", False))
        self._manning_normalized: torch.Tensor | None = None
        self.manning_stats: dict[str, float] | None = None
        if self.include_manning:
            lookup_config = dataset_config.get("manning_lookup") or DEFAULT_MANNING_LOOKUP
            lookup = {int(code): float(value) for code, value in dict(lookup_config).items()}
            fallback = float(dataset_config.get("manning_fallback", DEFAULT_MANNING_FALLBACK))
            self._load_manning(lookup, fallback)

    # ---------------------------------------------------------------- manning

    def _load_manning(self, lookup: dict[int, float], fallback: float) -> None:
        # self.event already holds the exact folder/file name ("Australia",
        # "UK", ...) via the EVENTS mapping -- do NOT re-capitalize ("UK"->"Uk").
        lulc_path = Path(self.root) / "Relevant data" / "Land use and land cover" / f"{self.event}.tif"
        if not lulc_path.exists():
            candidates = list((Path(self.root) / "Relevant data" / "Land use and land cover").glob("*.tif"))
            matches = [p for p in candidates if p.stem.lower() == self.event.lower()]
            if not matches:
                raise FileNotFoundError(f"LULC raster not found for event {self.event!r} under {lulc_path.parent}")
            lulc_path = matches[0]
        lulc = _read_raster(lulc_path)
        codes = torch.from_numpy(np.ascontiguousarray(lulc)).long()
        manning = torch.full(codes.shape, fallback, dtype=torch.float32)
        for code, value in lookup.items():
            manning[codes == code] = value
        # Categorical source -> NEAREST resize to the water grid (bilinear
        # would blend adjacent land-cover classes into meaningless codes).
        manning = F.interpolate(
            manning.unsqueeze(0).unsqueeze(0), size=(self.height, self.width), mode="nearest"
        )[0, 0]
        mean = float(manning.mean())
        std = float(manning.std().clamp(min=1e-6))
        self.manning_stats = {"mean": mean, "std": std, "lookup": {str(k): v for k, v in lookup.items()},
                              "fallback": fallback, "source": str(lulc_path)}
        self._manning_normalized = (manning - mean) / std

    # ------------------------------------------------------------ eval masks

    def _occupancy(self) -> torch.Tensor:
        """Train-split water-occupancy map (fraction of subsampled train
        frames wet at WET_OCCUPANCY_THRESHOLD_M), cached on disk keyed by
        event/fidelity/resolution."""

        if self._occupancy_map is not None:
            return self._occupancy_map
        cache_dir = Path(__file__).resolve().parents[1] / "outputs" / "floodcastbench_masks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"{self.event}_{self.fidelity}_{self.resolution}_g{WET_OCCUPANCY_THRESHOLD_M}_s{OCCUPANCY_FRAME_STRIDE}"
        cache_path = cache_dir / f"water_occupancy_{hashlib.md5(key.encode()).hexdigest()[:10]}.npz"
        if cache_path.exists():
            payload = np.load(cache_path)
            self._occupancy_map = torch.from_numpy(payload["occupancy"]).float()
            return self._occupancy_map
        train_start, train_end = split_frame_ranges(
            len(self.frames), self.config.get("dataset", {}).get("split_counts")
        )["train"]
        wet_sum = np.zeros((self.height, self.width), dtype=np.float64)
        count = 0
        for frame in self.frames[train_start:train_end:OCCUPANCY_FRAME_STRIDE]:
            depth = _read_raster(frame.path)
            wet_sum += (depth >= WET_OCCUPANCY_THRESHOLD_M).astype(np.float64)
            count += 1
        occupancy = (wet_sum / max(count, 1)).astype(np.float32)
        np.savez_compressed(cache_path, occupancy=occupancy, key=key)
        self._occupancy_map = torch.from_numpy(occupancy).float()
        return self._occupancy_map

    def _eval_mask(self, window_index: int, height: int, width: int) -> torch.Tensor:
        if self.eval_mask_structure == "random":
            return super()._eval_mask(window_index, height, width)
        bank_slot = window_index % max(self.eval_mask_bank_size, 1)
        cache_key = (self.eval_mask_structure, bank_slot)
        cached = self._eval_mask_bank.get(cache_key)
        if cached is not None and cached.shape[1:] == (height, width):
            return cached
        generator = torch.Generator().manual_seed(self.eval_mask_seed + bank_slot)
        if self.eval_mask_structure == "gauge":
            mask = generate_gauge_mask(height, width, self.missing_rate, self._occupancy(), generator=generator)
        else:
            mask = generate_cluster_mask(height, width, self.missing_rate, generator=generator)
        self._eval_mask_bank[cache_key] = mask
        return mask

    # ---------------------------------------------------------------- samples

    def _read_water_patch(self, path: Path, y0: int, x0: int, height: int, width: int) -> np.ndarray:
        # Overrides V1's frozen version (which windowed-reads straight from
        # disk every call) when dataset.cache_frames_in_memory is set --
        # see the flag's docstring in __init__. Decodes each distinct frame
        # exactly once per worker process, then slices the patch from the
        # in-memory array; numerically identical to V1's windowed read (same
        # source pixels), just without repeating the disk decode.
        if not self.cache_frames_in_memory:
            return super()._read_water_patch(path, y0, x0, height, width)
        cached = self._water_frame_cache.get(path)
        if cached is None:
            cached = _read_raster(path)
            self._water_frame_cache[path] = cached
        if height == self.height and width == self.width:
            return cached
        return cached[y0 : y0 + height, x0 : x0 + width]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.missing_rate_range is not None:
            low, high = self.missing_rate_range
            self.missing_rate = low + (high - low) * float(torch.rand(1).item())
        if self.frame_stride != 1:
            # See _StridedFrameView docstring: temporarily swap self.frames
            # for the duration of V1's (frozen, unmodified) __getitem__ call
            # so its internal `self.frames[start:start+window_length]` slice
            # transparently returns frame_stride-spaced frames instead of
            # consecutive ones. Restored in `finally` regardless of outcome.
            original_frames = self.frames
            self.frames = _StridedFrameView(original_frames, self.frame_stride)
            try:
                sample = super().__getitem__(index)
            finally:
                self.frames = original_frames
        else:
            sample = super().__getitem__(index)
        if self.include_manning:
            y0, x0 = sample["meta"]["patch_origin"]
            ph, pw = sample["meta"]["patch_size"]
            sample["manning"] = self._manning_normalized[y0 : y0 + ph, x0 : x0 + pw].unsqueeze(0).clone()
        if self.augmentation:
            transform_index = int(torch.randint(0, 8, (1,)).item())
            if transform_index:
                keys = _SPATIAL_KEYS + (("manning",) if self.include_manning else ())
                for key in keys:
                    sample[key] = apply_dihedral(sample[key], transform_index)
            sample["meta"]["augmentation_dihedral"] = transform_index
        return sample


def build_diff_sparse_v2_dataset(
    root: str | Path,
    config: dict[str, Any],
    split: str = "train",
    normalization_stats: dict[str, Any] | None = None,
    patch_mode: str | None = None,
) -> FloodCastBenchDiffSparseV2Dataset:
    return FloodCastBenchDiffSparseV2Dataset(
        root,
        config,
        split=split,
        normalization_stats=normalization_stats,
        patch_mode=patch_mode,
    )
