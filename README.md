## CROWD-city

## Citation and usage of code
If you use this work for academic work please cite the following paper:

> Alam, M. S., Martens, M. H., & Bazilinskyy, P. (2025). 

The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code. For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

## Getting started
[![Python Version](https://img.shields.io/badge/python-3.10.18-blue.svg)](https://www.python.org/downloads/release/python-31018/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.10.18** and the [`uv`](https://docs.astral.sh/uv/) package manager.
Follow these steps to set up the project.

**Step 1:** Install `uv`. `uv` is a fast Python package and environment manager. Install it using one of the following methods:

**macOS / Linux (bash/zsh):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (if you already have Python and pip):**
```bash
pip install uv
```

**Step 2:** Fix permissions (if needed):

Sometimes `uv` needs to create a folder under `~/.local/share/uv/python` (macOS/Linux) or `%LOCALAPPDATA%\uv\python` (Windows).
If this folder was created by another tool (e.g. `sudo`), you may see an error like:
```lua
error: failed to create directory ... Permission denied (os error 13)
```

To fix it, ensure you own the directory:

### macOS / Linux
```bash
mkdir -p ~/.local/share/uv
chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv
chmod -R u+rwX ~/.local/share/uv
```

### Windows
```powershell
# Create directory if it doesn't exist
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\uv"

# Ensure you (the current user) own it
# (usually not needed, but if permissions are broken)
icacls "$env:LOCALAPPDATA\uv" /grant "$($env:UserName):(OI)(CI)F"
```

**Step 3:** After installing, verify:
```bash
uv --version
```

**Step 4:** Clone the repository:
```command line
git clone https://github.com/crowd-dataset/crowd-city
cd crowd-city
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.10.18 installed, let `uv` fetch it:
```command line
uv python install 3.10.18
```
`pyproject.toml` pins `requires-python = "==3.10.18"`, so `uv` uses this version.

**Step 6:** Create and sync the virtual environment. This will create **.venv** in the project folder and install dependencies exactly as locked in **uv.lock**:
```command line
uv sync --frozen
```

**Step 7:** Activate the virtual environment:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
.\.venv\Scripts\activate.bat
```

**Step 8:** Make sure the datasets are available: `mapping.csv` in the project root (or wherever `mapping` points), and the detection data in the directories set by `data` and `parquet_data` in `config`.


**Step 9:** Run the code:
```command line
python3 analysis.py
```

### Configuration of project
Configuration of the project is defined in `config`. Use `default.config` for the required structure: any setting missing from `config` that is marked optional in `common.py` falls back to its value in `default.config`. The config file has the following parameters (the segmentation settings are described in the next section):
- **`data`**: List of directories holding the YOLO detection output; the detection CSV files are read from their `bbox/` subfolder.
- **`parquet_data`**: List of directories holding the Parquet copy of the detections, one per entry in `data` and in the same order. The analysis reads detections only from here (`<root>/bbox/*.parquet`).
- **`sync_parquet_on_start`**: When `true`, new or changed CSV files under `data` are converted into the Parquet store before the analysis starts.
- **`videos`**: Directories containing the videos used to generate the data.
- **`mapping`**: CSV file with the city metadata and the list of videos and segments for each city.
- **`always_analyse`**: Always recompute the analysis, even when a matching `results.pickle` exists (useful for testing).
- **`waymo_dataset_path`**: Location of the raw Waymo Open Dataset used to calibrate the crossing detector and the metric speed model.
- **`process_waymo_if_missing`**: When `true`, the processed Waymo data is generated from `waymo_dataset_path` if it does not exist yet.
- **`min_max_videos`**: Number of fastest and slowest crossings for which video snippets are produced; `0` disables this.
- **`bbox_tracker`**: Tracker configuration file for YOLO tracking, used by `helper_script.py`.
- **`cpu_worker`**: Number of worker processes used to analyse detection files in parallel (also overridable with the `CROWD_CSV_WORKERS` environment variable).
- **`reanalyse_waiting_time`**: Recompute the crossing initiation time aggregates from the cached per-track values, for example after changing `min_waiting_time` or `max_waiting_time`.
- **`min_waiting_time`**: Minimum crossing initiation time, in seconds, for a crossing to be included.
- **`max_waiting_time`**: Maximum crossing initiation time, in seconds, for a crossing to be included.
- **`min_locality_population_percentage`**: A city is also kept when its population is at least this fraction of its country's population, even if it is below `population_threshold`.
- **`check_per_sec_time`**: Number of position checks per second used when measuring how long a pedestrian stands still before crossing.
- **`analysis_level`**: Level at which results are reported: `city` or `country`.
- **`boundary_left`**: x-coordinate of one edge of the crossing area used to detect road crossings (normalised between 0 and 1).
- **`boundary_right`**: x-coordinate of the opposite edge of the crossing area used to detect road crossings (normalised between 0 and 1).
- **`population_threshold`**: Minimum city population for a city to be included in the analysis.
- **`footage_threshold`**: Minimum total footage, in seconds, for a city to be included in the analysis.
- **`min_crossing_detect`**: Minimum number of detected crossings for a country or city to be kept in the output; `0` disables this filter.
- **`reanalyse_speed`**: Recompute the crossing speeds from the cached per-track values, for example after changing `min_speed_limit` or `max_speed_limit`, or when the installed speed model reports a different unit from the cached results.
- **`min_speed_limit`**: Minimum crossing speed for a crossing to be included.
- **`max_speed_limit`**: Maximum crossing speed for a crossing to be included.
- **`countries_analyse`**: ISO3 codes of the countries to analyse; an empty list analyses all countries.
- **`n_cities`**: Number of cities to analyse, chosen by total footage: a positive value keeps the cities with the most footage, a negative value those with the least, and `null` keeps all cities.
- **`max_footage_hours_per_city`**: Caps the footage analysed per city, in hours; `null` analyses everything. Segments are drawn in a random order per city rather than in mapping order, so the budget is spread across that city's videos. Segments with no detection file in the Parquet store, or with a vehicle type outside `vehicles_analyse`, are skipped and the next segment is drawn in their place, so the whole budget goes to footage that is actually analysed. The last segment drawn is trimmed to fit the cap.
- **`footage_sampling_seed`**: Seed for that random draw (default `42`). The same seed always selects the same segments, so runs are reproducible; change it to analyse a different sample.
- **`processing_fps`**: Frame rate the detections are resampled to before analysis; `null` keeps each video's own frame rate.
- **`vehicles_analyse`**: Vehicle types (the codes in the mapping's `vehicle_type` column) to analyse; an empty list analyses footage from all vehicle types.
- **`min_confidence`**: Minimum YOLO detection confidence for a detection to be used.
- **`font_family`**: Font family used in the figures.
- **`font_size`**: Font size used in the figures.
- **`plotly_template`**: Plotly template used for the figures.
- **`logger_level`**: Level of console output: `debug`, `info`, `warning` or `error`.
- **`ftp_base_url`**: Base URL of the file server that hosts the videos. Used by the segmentation pass, the segmentation sample renderer and the crossing validation tool.
- **`display_frame_tracking`**: Read by `helper_script.py` but currently has no effect.
- **`save_annoted_img`**: Read by `helper_script.py` but currently has no effect.
- **`save_tracked_img`**: Read by `helper_script.py` but currently has no effect.
- **`delete_labels`**: Read by `helper_script.py` but currently has no effect.
- **`delete_frames`**: Read by `helper_script.py` but currently has no effect.
- **`save_images`**: Whether to export PNG and EPS alongside the interactive HTML. Raster export goes through kaleido, which launches a headless Chromium and is prone to hanging indefinitely on Windows. Set this to `false` to keep only the HTML, which carries the same data, when a run stalls on "Saving png file for ...".

### Road-surface segmentation
Crossing speed and crossing initiation time can additionally be measured against the road surface rather than from bounding-box motion alone. When enabled, the analysis segments only the video windows that contain an already-detected crossing with SegFormer finetuned on Cityscapes, reads the surface under each pedestrian's feet, and derives the interval that pedestrian actually spends on the carriageway. The initiation time then becomes the stationary interval immediately before stepping onto the road, and the speed is fitted over the on-road frames only.

The results are written to separate `speed_crossing_seg_*` and `time_crossing_seg_*` columns. The baseline `speed_crossing_*` and `time_crossing_*` columns are left untouched, so the two derivations can be compared before either is preferred. Note that the frozen Waymo speed model was calibrated on whole algorithm-selected tracks, so restricting its input window moves it away from the distribution it was validated on; the count of tracks rejected as `out_of_calibration_distribution` is logged for exactly this reason.

Nothing needs a surface label for every frame. The crossing speed is measured between road entry and road exit, and the initiation time is the wait ending at road entry, so only those two moments have to be located. A coarse pass therefore establishes the shape of each track, and a finer pass looks only inside the spans that must contain a change. Tracks crossing at the same moment share the decoded frames, and a track already on the carriageway throughout costs nothing beyond the coarse pass.

Video is never downloaded in full: ffmpeg seeks over HTTP range requests and decodes only the crossing windows. Only the derived per-frame surface labels are cached, keyed by a fingerprint of the crossing configuration, so re-tuning the crossing detector correctly invalidates the store rather than reusing labels sampled for a different set of tracks.

- **`use_segmentation`**: Enables the road-surface segmentation pass. When disabled, the analysis is exactly the baseline.
- **`segmentation_is_primary`**: Determines which derivation the figures report. When `false` (the default) every figure and correlation keeps using the baseline bounding-box metrics and the segmentation values are written only to the `speed_crossing_seg_*` and `time_crossing_seg_*` columns for comparison. When `true` the segmentation values become the reported crossing speed and initiation time throughout. Note that the segmentation metrics do not cover every crossing the baseline covers: a track is dropped when it never reaches the carriageway, when the frozen speed model's reliability gates reject the shortened window, and, for the initiation time, whenever the pedestrian was already on the road when first detected. The count of localities that lose a value is logged when this is enabled.
- **`seg_data`**: Directory holding the persistent surface-label store.
- **`segmentation_model`**: Hugging Face identifier of the SegFormer Cityscapes checkpoint. Any `nvidia/segformer-b*-finetuned-cityscapes` checkpoint works; `b4` and `b5` are more accurate and slower, `b0` is the escape hatch when throughput rather than accuracy is the binding constraint. Segmentation is the dominant cost of this pass, so run it on a CUDA device.
- **`segmentation_device`**: Torch device for segmentation; `auto` prefers CUDA, then MPS, then CPU.
- **`segmentation_batch_size`**: Number of frames passed through the model at once.
- **`segmentation_coarse_hz`**: Sampling rate of the first pass, which only has to establish the shape of each track: off the carriageway, then on it, then off it again.
- **`segmentation_refine_hz`**: Sampling rate of the second pass, which looks only inside the short spans where a transition must lie. This sets how precisely the start of the crossing is located, and therefore the resolution of the initiation time.
- **`segmentation_input_width`** / **`segmentation_input_height`**: Resolution frames are scaled to before segmentation.
- **`segmentation_min_confidence`**: Minimum per-pixel confidence for a pixel to contribute to the surface decision.

The file server credentials (`ftp_username`, `ftp_password`, `ftp_token`) in `secret` are required for this pass, because the crossing windows are read from the server.

For working with external APIs of [VideoFiles](https://files.mobility-squad.com/), [GeoNames](https://www.geonames.org), [BEA](https://apps.bea.gov/api/signup), [TomTom](https://developer.tomtom.com/user/register), [Trafikab](https://www.trafiklab.se/api/trafiklab-apis), and [Numbeo](https://www.numbeo.com/common/api.jsp) (paid), the API keys need to be placed in file `secret` (no extension) in the root of the project. The file needs to be formatted as `default.secret`. These keys are optional for just running the analysis on the dataset, except for the file server credentials needed by the segmentation pass.


## Contact
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com or pavlo.bazilinskyy@gmail.com.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
