## CROWD-city

Analyses how pedestrians cross the road in dashcam footage from cities around the world (the [CROWD dataset](https://github.com/crowd-dataset/crowd)). For every city it counts road crossings and measures crossing speed in m/s and crossing initiation time (how long a pedestrian stands still before stepping onto the road), then relates them to city and country indicators.

![Cities in the analysis](docs/images/cities_world_map.png)
*Cities in the current analysis (red dots) and the countries they are in.*

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
Configuration of the project is defined in `config`. Every value is read from `config`; `default.config` is only a template listing the settings that must exist. Its values are never used, so if `config` is missing any setting the run stops and names it. The config file has the following parameters (the segmentation settings are described in the next section):
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
- **`yolo_imgsz`**: Input size in pixels YOLO resizes each frame to when tracking the Waymo calibration videos (640). Each tracking CSV records the size it was made with, and changing this re-tracks the videos and rebuilds the speed model, since a speed model only fits tracks made with the same settings. 1280 finds more small, distant pedestrians, but its speed model failed the untouched validation test, so 640 is used.
- **`cpu_worker`**: Number of worker processes used to analyse detection files in parallel (also overridable with the `CROWD_CSV_WORKERS` environment variable).
- **`reanalyse_waiting_time`**: Recompute the crossing initiation time aggregates from the cached per-track values, for example after changing `min_waiting_time` or `max_waiting_time`.
- **`min_waiting_time`**: Minimum crossing initiation time, in seconds, for a crossing to be included.
- **`max_waiting_time`**: Maximum crossing initiation time, in seconds, for a crossing to be included.
- **`min_locality_population_percentage`**: A city is also kept when its population is at least this fraction of its country's population, even if it is below `population_threshold`.
- **`check_per_sec_time`**: Number of position checks per second used when measuring how long a pedestrian stands still before crossing. The checks are `round(fps / check_per_sec_time)` frames apart, and a check counts as standing still when the pedestrian moves no more than 10% of their box height. A wait needs at least three consecutive still checks, so the shortest recorded wait is about `3 / check_per_sec_time` seconds. Both the bounding-box and the road-surface initiation times are measured from the frames the checks actually span, so they stay in true seconds at any frame rate (at 40 fps a check is 13 frames, i.e. 0.325 s). A higher value lets slower walking pass as standing still, because the 10% limit applies to each check.
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
- **`cities_analyse`**: Cities to analyse; an empty list analyses all cities. City names repeat across countries and states, so each entry is `"City"`, `"City, ISO3"` or `"City, State, ISO3"`, for example `["Aberdeen, GBR", "Aberdeen, WA, USA", "Tokyo"]`. State and ISO3 are written as in `mapping.csv`, and names are matched case-insensitively against `locality` and `locality_aka`. A bare name keeps every city with that name and logs which ones matched. An entry that matches no city in `mapping.csv` stops the run. When `countries_analyse` is also set, a city must satisfy both lists, and `n_cities` then chooses among the cities that remain.
- **`n_cities`**: Number of cities to analyse, chosen by total footage: a positive value keeps the cities with the most footage, a negative value those with the least, and `null` keeps all cities.
- **`max_footage_hours_per_city`**: Caps the footage analysed per city, in hours; `null` analyses everything. Segments are drawn in a random order per city rather than in mapping order, so the budget is spread across that city's videos. Segments with no detection file in the Parquet store, or with a vehicle type outside `vehicles_analyse`, are skipped and the next segment is drawn in their place, so the whole budget goes to footage that is actually analysed. The last segment drawn is trimmed to fit the cap.
- **`footage_sampling_seed`**: Seed for that random draw (default `42`). The same seed always selects the same segments, so runs are reproducible; change it to analyse a different sample.
- **`target_crossings_per_city`**: Number of pedestrian crossings to collect per city, or `null` to switch this off (the default). When set, each city's footage is processed in the same seeded random order as `max_footage_hours_per_city`, one segment per city per round, and a city stops receiving footage once it has at least this many crossings (as counted by `crossing_rule`). Whole segments are processed, so a city can end slightly above the target; a city that runs out of footage keeps what it found. When `max_footage_hours_per_city` is also set, whichever limit is reached first applies. Footage time, object counts and every other statistic are then taken over exactly the footage processed. With `road_crossing`, the limit on unreadable segments applies to the whole run rather than to each round.
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

### Choosing which cities to analyse
The analysis runs on every city in `mapping.csv` unless you narrow it down. The settings take effect in this order:

1. `countries_analyse` keeps only the listed countries (ISO3 codes).
2. `cities_analyse` keeps only the listed cities.
3. `n_cities` keeps the cities with the most footage (or the least, if negative) out of those that remain.
4. `max_footage_hours_per_city` and `target_crossings_per_city` limit how much of each remaining city's footage is processed.
5. After the detections are analysed, `population_threshold`, `min_locality_population_percentage` and `footage_threshold` drop small cities and cities with little footage from the reported results.

City names are not unique: `mapping.csv` has, for example, an Aberdeen in Great Britain and two in the United States. Each `cities_analyse` entry can therefore name the country, and the state too when a country has several cities with that name:

```json
"countries_analyse": [],
"cities_analyse": [
  "Aberdeen, GBR",
  "Aberdeen, WA, USA",
  "Tokyo"
],
```

- `"City, ISO3"` picks the city in one country, and `"City, State, ISO3"` picks one within a country. Write state and ISO3 exactly as in the `state` and `iso3` columns of `mapping.csv`.
- A bare `"City"` keeps every city with that name. The log then lists the cities it matched, so you can narrow the entry down if that was not intended.
- Case is ignored, and alternative names in `locality_aka` are matched too, so `"Aliabad-e Katul"` finds Ali Abad in Iran.
- An entry that matches no city in `mapping.csv`, such as a misspelling, stops the run at the start with an error naming it, instead of silently analysing nothing. A listed city that the thresholds in step 5 later remove is only noted in the log.
- Leave both lists empty (`[]`) to analyse all cities.

Changing either list changes which results are valid, so the next run reanalyses instead of reusing `results.pickle`. With `always_analyse` set to `false`, a rerun with unchanged settings reuses `results.pickle` and goes straight to the figures.

### Road-surface segmentation
Crossing speed and crossing initiation time can additionally be measured against the road surface rather than from bounding-box motion alone. When enabled, the analysis segments only the video windows that contain an already-detected crossing with SegFormer finetuned on Cityscapes, reads the surface under each pedestrian's feet, and derives the interval that pedestrian actually spends on the carriageway. The initiation time then becomes the stationary interval immediately before stepping onto the road, and the speed is fitted over the on-road frames only.

The mapping output keeps both derivations side by side: `speed_crossing_bbox_*` and `time_crossing_bbox_*` always hold the bounding-box values, `speed_crossing_seg_*` and `time_crossing_seg_*` always hold the segmentation values, and `speed_crossing_*` and `time_crossing_*` hold whichever one `segmentation_is_primary` selects for reporting, which is what every figure reads. Note that the frozen Waymo speed model was calibrated on whole algorithm-selected tracks, so restricting its input window moves it away from the distribution it was validated on; the count of tracks rejected as `out_of_calibration_distribution` is logged for exactly this reason.

Nothing needs a surface label for every frame. The crossing speed is measured between road entry and road exit, and the initiation time is the wait ending at road entry, so only those two moments have to be located. A pedestrian who steps onto the road without first standing still for at least three position checks (see `check_per_sec_time`) has no initiation time and is left out of the averages rather than counted as zero, as in the bounding-box metric; so is one who is already on the road when first seen. A coarse pass therefore establishes the shape of each track, and a finer pass looks only inside the spans that must contain a change. Tracks crossing at the same moment share the decoded frames, and a track already on the carriageway throughout costs nothing beyond the coarse pass.

Video is never downloaded in full: ffmpeg seeks over HTTP range requests and decodes only the crossing windows. Only the derived per-frame surface labels are cached, keyed by a fingerprint of the crossing configuration, so re-tuning the crossing detector correctly invalidates the store rather than reusing labels sampled for a different set of tracks.

**What it looks like.** Each sample below shows the road (red) and footpath (green) found by the segmentation model, and a box on each pedestrian counted as crossing. For these images the boxes were also labelled with each pedestrian's road-restricted speed and initiation time. `speed: n/a` means the speed model's reliability gates rejected that track; `wait unseen` means the pedestrian was already on the road when first seen.

![Crossings that are counted correctly](docs/images/crossing_samples_good.jpg)
*Correctly counted crossings: a zebra crossing in Catania (1.39 m/s, no wait), a signalised crossing in Birmingham, a pedestrian waiting 2.56 s at the kerb in Prague at night, and a crossing in front of a shop in the United States.*

Earlier versions of the rule also counted some pedestrians who were not crossing; the common causes were snow at the road edge labelled as road, and people walking along the edge of the carriageway. The current rule requires the pedestrian to pass in front of the camera and to walk across, which removes these cases (see [Hand-verified precision test](#hand-verified-precision-test)):

![Typical false crossings](docs/images/crossing_samples_failures.jpg)
*False crossings of an earlier rule version: in Sapporo the snowbank is labelled as road, so a pedestrian beside it counts as on the road; in Bengaluru a pedestrian walks along the road edge beside the barrier.*

To render sample clips from your own run, with the surface overlay and the boxes labelled by track id (the clips are written to `_output/segmentation_samples/`):

```bash
uv run python visualize_segmentation_samples.py --samples 8
```

- **`crossing_rule`**: Decides which pedestrians count as crossing, for every count, speed, waiting time and figure. `detector` uses the CROWD crossing detector alone: the track must pass through the vertical strip between `boundary_left` and `boundary_right` and survive its motion filters. `road_crossing` uses the road surface instead, and is tuned for precision first: whenever it counts a crossing, that should really be a pedestrian walking across the road in front of the camera. Broken YOLO tracks of sideways-walking pedestrians are first joined (`utils/crossing/track_joining.py`); a pedestrian then counts only when the track passes the centre of the image (in front of the camera), is not a cyclist or rider, moves independently of the camera, has its feet on the road while moving across at least `min_crossing_x_range` of the image with a slowly changing box size (which rejects people walking along the road), and walks at a plausible pace that does not speed up sharply. On every reviewed test, Waymo and hand-checked CROWD footage, its precision is 100%, at a recall of about a third (see [Hand-verified precision test](#hand-verified-precision-test)). It requires `use_segmentation` and the video file server: candidates are first found from the boxes alone, then segmented, and the run stops if the road surface cannot be read for more than 5% of segments rather than undercounting crossings. The rules are defined in `utils/crossing/road_crossing.py`.
- **`use_segmentation`**: Enables the road-surface segmentation pass. When disabled, the analysis is exactly the baseline. Required when `crossing_rule` is `road_crossing`.
- **`segmentation_is_primary`**: Determines which derivation the figures report. When `false` every figure and correlation uses the bounding-box metrics. When `true` the segmentation values become the reported crossing speed and initiation time throughout, i.e. in `speed_crossing_*` and `time_crossing_*`. Either way the bounding-box values stay in `speed_crossing_bbox_*` / `time_crossing_bbox_*` and the segmentation values in `speed_crossing_seg_*` / `time_crossing_seg_*`, so the two can always be compared. Note that the segmentation metrics do not cover every crossing the baseline covers: a track is dropped when it never reaches the carriageway, when the frozen speed model's reliability gates reject the shortened window, and, for the initiation time, whenever the pedestrian was already on the road when first detected. The count of localities that lose a value is logged when this is enabled.
- **`seg_data`**: Directory holding the persistent surface-label store.
- **`segmentation_model`**: Hugging Face identifier of the SegFormer Cityscapes checkpoint. Any `nvidia/segformer-b*-finetuned-cityscapes` checkpoint works; `b4` and `b5` are more accurate and slower, `b0` is the escape hatch when throughput rather than accuracy is the binding constraint. Segmentation is the dominant cost of this pass, so run it on a CUDA device.
- **`segmentation_device`**: Torch device for segmentation; `auto` prefers CUDA, then MPS, then CPU.
- **`segmentation_parallel_videos`**: Number of detection segments whose video windows are read from the file server at once. Reading is almost entirely waiting on the server, so a few in parallel cut the run time roughly proportionally, while the GPU model still runs one batch at a time. Failed reads (the server returns transient 5XX errors under load) are retried three times with growing pauses, and a segment that still cannot be read counts as failed rather than being stored with missing frames. Keep this modest (about 3), since the server struggles under many concurrent requests.
- **`segmentation_batch_size`**: Number of frames passed through the model at once.
- **`segmentation_coarse_hz`**: Sampling rate of the first pass, which only has to establish the shape of each track: off the carriageway, then on it, then off it again.
- **`segmentation_refine_hz`**: Sampling rate of the second pass, which looks only inside the short spans where a transition must lie. This sets how precisely the start of the crossing is located, and therefore the resolution of the initiation time.
- **`segmentation_input_width`** / **`segmentation_input_height`**: Resolution frames are scaled to before segmentation.
- **`segmentation_min_confidence`**: Minimum per-pixel confidence for a pixel to contribute to the surface decision.

The file server credentials (`ftp_username`, `ftp_password`, `ftp_token`) in `secret` are required for this pass, because the crossing windows are read from the server.

For working with external APIs of [VideoFiles](https://files.mobility-squad.com/), [GeoNames](https://www.geonames.org), [BEA](https://apps.bea.gov/api/signup), [TomTom](https://developer.tomtom.com/user/register), [Trafikab](https://www.trafiklab.se/api/trafiklab-apis), and [Numbeo](https://www.numbeo.com/common/api.jsp) (paid), the API keys need to be placed in file `secret` (no extension) in the root of the project. The file needs to be formatted as `default.secret`. These keys are optional for just running the analysis on the dataset, except for the file server credentials needed by the segmentation pass.


### Checking crossings by hand
`crossing_review.py` is a small local web tool for counting, on real footage, how many pedestrians actually cross and how many of them the pipeline gets right. Start it and open the address it prints:

```bash
uv run python crossing_review.py
```

Enter a video id and press **Process video**. This does all the slow work once, before you start, so the review itself has no lag: the whole video is downloaded (over 8 parallel connections) into `_output/crossing_review/videos/`, every segment's YOLO person tracks are read (detection files missing locally are fetched from the file server's `data` alias, see `--csv-url-path`, and converted to Parquet), and the crossing algorithm, i.e. the detection worker and then `crossing_rule` with the current `config`, runs on every full segment, reading the local video. An hour of video takes roughly 10 minutes, mostly the download.

Then pick a segment. The video plays from the local copy, with the crossings the algorithm counted drawn as **yellow** boxes:

- **Click a yellow box if it is a real crossing**; it turns **green**.
- **Click every other pedestrian you see crossing.** If YOLO detected them, a **purple** box appears around them; if not, an **orange** circle marks the spot.
- Click a green or purple box, or an orange circle, again to undo.

The four counts update as you go:

1. **Crossing, not detected by YOLO**: orange circles.
2. **Crossing detected by YOLO, missed by the algorithm**: purple boxes.
3. **Crossing detected by YOLO and counted by the algorithm**: green boxes.
4. **Fake crossing**: yellow boxes never confirmed. They are listed with a **Go** button, so you can check each before finishing.

The page shows them per segment or for the whole video, with precision (3 / (3 + 4)), recall (3 / (1 + 2 + 3)) and the share YOLO detected. Mark a segment fully reviewed once you have watched all of it. Labels are saved after every change in `_output/crossing_review/labels/<video_id>.json`, and **Export CSV** writes the counts per segment. The file-server credentials come from the secrets file and never reach the browser.

### Hand-verified precision test
The crossing algorithm is held to **100% precision**: every pedestrian it counts as crossing must really walk across the road in front of the camera. Feet on the road alone, walking along the road, or crossing a side street do not count. Recall is secondary. Precision is checked by hand with `crossing_review.py` on a fixed set of about one hour of daytime footage from a car in each of five cities, listed with their video ids in [`crossing_review_set.json`](crossing_review_set.json). These videos were not used to tune the rule, so they are an honest test.

| City | Video id | YouTube URL | Footage | Counted by the algorithm | Real (confirmed) | Fake | Precision | Missed by the algorithm (YOLO detected) | Not detected by YOLO | Recall |
|---|---|---|---|---|---|---|---|---|---|---|
| Los Angeles | `1LS7MhOyhro` | <https://www.youtube.com/watch?v=1LS7MhOyhro> | 60.0 min from 0 s | 5 | 5 | 0 | **100%** | 7 | 3 | 5 / 15 (33%) |
| Amsterdam | `iVJGEW1st8c` | <https://www.youtube.com/watch?v=iVJGEW1st8c> | 59.5 min from 0 s | 25 |  |  | pending review |  |  |  |
| Seoul | `XuYX93xqjB4` | <https://www.youtube.com/watch?v=XuYX93xqjB4&t=10s> | 62.4 min from 10 s | 25 |  |  | pending review |  |  |  |
| Sydney | `u084OpLn2Ps` | <https://www.youtube.com/watch?v=u084OpLn2Ps&t=38s> | 59.7 min from 38 s | 2 |  |  | pending review |  |  |  |
| Cairo | `a4zcL56YSME` | <https://www.youtube.com/watch?v=a4zcL56YSME&t=27s> | 73.0 min from 27 s | 0 | 0 | 0 | — (nothing counted) |  |  |  |

In Los Angeles, YOLO detected 12 of the 15 pedestrians who crossed, and the algorithm counted 5 of those 12. The first Sydney video chosen (`JMUROQ59kAQ`) was replaced because its detection file on the file server holds only the first second.

The rule was tuned on other data, which is reported separately:

| Data | Counted | Fake | Precision | Recall |
|---|---|---|---|---|
| Waymo training (798 recordings; tuning) | 205 | 0 | 100% | 140 / 416 crosswalk crossers (34%) |
| Waymo validation (202 recordings; held out) | 39 | 0 | 100% | 24 / 79 crosswalk crossers (30%) |
| Paris, video id `AdqE7mFQ7Y4` (<https://www.youtube.com/watch?v=AdqE7mFQ7Y4&t=31s>), first 12 minutes from 31 s (busy; used while tuning) | 22 | 0 | 100% | about 23 of 54 crossers (43%) |

On Waymo, a counted pedestrian is real when Waymo labels them as crossing on a crosswalk or, since those labels cover marked crosswalks only, when the hand review of the clip confirms a crossing elsewhere (62 in training, 15 in validation).

**Trying to count more of the missed crossers.** Of the 7 Los Angeles crossers YOLO detected but the algorithm missed, 1 is a distant figure (box 0.07 of the image high) seen for under 2 s near the centre, 3 are only tracked on one side of the centre, so they never pass in front of the camera within their track, 1 fails both the box-size and the camera-motion checks, and 2 are first seen inside the centre strip and walk out to the left (one of them also moves with the camera). The only change that would recover any of them is to also count pedestrians first seen *inside* the centre strip who then walk out to one side, such as someone stepping out from behind a vehicle in front of the camera. It counts 1 more real crosser in Los Angeles, 2 more in Paris and 19 more crosswalk crossers in Waymo training, but it also counts 2 pedestrians in Paris that the review marked as not crossing, plus 18 unreviewed Waymo picks. It was therefore not adopted: precision comes first.

### Waymo calibration: current results and what was tried
Crossing speeds are reported in m/s by a speed model calibrated on the [Waymo Open Dataset](https://waymo.com/open/), whose lidar-derived pedestrian speeds serve as the reference. The analysis refuses to run without a qualified model rather than falling back to a relative index. When `process_waymo_if_missing` is enabled, the raw Waymo TFRecords are exported (in Docker if available, otherwise in a local `uv` environment), tracked with YOLO and BoT-SORT at `yolo_imgsz`, and the speed model is fitted on the Waymo training split and tested once on the untouched validation split. The model is only used when it passes both the cross-validation and the external validation checks.

**Current result** (640 px tracking, speed model trained on the CROWD detector's crossings):

| | Tracks | MAE | RMSE | Bias | Within 0.50 m/s |
|---|---|---|---|---|---|
| Development, source-grouped cross-validation | 166 | 0.19 m/s | 0.26 m/s | −0.02 m/s | 95.18% |
| Untouched validation | 27 | 0.13 m/s | 0.21 m/s | −0.05 m/s | 96.30% |

City averages are close to unbiased; individual crossings are typically off by 0.1–0.2 m/s, and estimates are compressed towards the mean (slow walkers read too fast, fast walkers too slow), so differences between cities are understated but their order is kept. With 27 validation tracks the validation MAE carries an uncertainty of about ±0.03–0.04 m/s.

![Waymo reference speed against estimated speed](docs/images/waymo_speed_validation.png)
*Estimated crossing speed against Waymo's lidar-derived reference speed, for the training fit, the source-held-out cross-validation and the untouched validation set. Points on the dashed line are exact.*

**Crossing detection**, scored against Waymo's crosswalk-crossing labels (which cover marked crosswalks only, so precision is a lower bound; hand audits found about half of the "wrong" picks to be real crossings elsewhere):

| Rule | Recall, training | Recall, validation | Precision (lower bound) |
|---|---|---|---|
| CROWD detector | 139 / 918 (15%) | 19 / 183 (10%) | 65% / 58% |
| Detector with feet on the road | 139 / 918 (15%) | 19 / 183 (10%) | 70% / 59% |
| `road_crossing`, first version (feet on the road only) | 232 / 918 (25%) | 51 / 183 (28%) | 49% / 49% |

These were the first comparisons. The `road_crossing` rule used now adds the passes-the-camera, rider, camera-motion and walking checks, and track joining; with Waymo's labels completed by hand review its precision is 100% on both splits (see [Hand-verified precision test](#hand-verified-precision-test)). About three quarters of the real crossers that are missed are never detected by YOLO at all (small, distant pedestrians). On the validation split, speeds of `road_crossing` crossings have an MAE of 0.15 m/s, against 0.11 m/s for the detector's crossings.

**What was tried**, all on the Waymo training split first and confirmed on the untouched validation split only once:

| Change | Result | Kept |
|---|---|---|
| Other speed models (random forest, gradient boosting, ridge, a single physics feature) | MAE 0.176–0.191 m/s vs 0.187 m/s; the model type is not the bottleneck | No |
| More training data of the same kind | Training on 25% of recordings scored the same as 100% | — |
| Stricter reliability gates (longer, steadier tracks; camera nearly still) | Error 15–25% lower, but a third to a half fewer crossings get a speed | No (trade-off) |
| Broad evaluation on every laterally moving pedestrian (`utils/crossing/waymo_broad_evaluation.py`) | The model is weak outside crossing-like tracks (r ≈ 0.2), so crossing selection is essential | Evaluation tool only |
| YOLO input 1280 px instead of 640 px | 40% more real crossers matched and 12% more found, but the speed model's validation spread fell to 0.49 (limit 0.65) | No |
| Worst-source gate judged only on sources with 3+ reliable tracks | A single well-detected runner no longer blocks every model; 640 px qualifies under both versions | Yes |
| Road-surface speed (feet-on-road frames only) | Same accuracy as whole-track speed (0.138 vs 0.138 m/s on the same crossers) | No |
| Road-surface crossing rules (`utils/crossing/waymo_segmentation_evaluation.py`) | `road_crossing` finds about twice as many real crossings | Yes, for counting |
| Training the speed model on the feet-on-road crossings | Validation spread fell to 0.59 (limit 0.65) | No |
| Camera motion from background optical flow instead of other objects' boxes | Lower-half points: −0.008 to −0.011 m/s, 95% interval includes zero; road-surface points: no gain | No |
| Speed-feature windows defined in seconds instead of frames (smoothing, outlier cleaning, local-rate and reversal steps, camera-motion pairing), keeping every frame at the video's own frame rate | Identical features at Waymo's 10 fps; on real 30 fps CROWD tracks the 30 vs 10 fps mismatch of direction reversals halves (51% to 23%) and 23% more tracks get an agreeing speed at both rates | Yes |

The remaining error comes mainly from the bounding boxes themselves (jitter, partial occlusion, and box height as a stand-in for distance). Further gains would likely need better boxes or a direct distance estimate, such as a monocular depth model, rather than parameter tuning.


## Example results
These figures come from a run with `max_footage_hours_per_city` set to 1, `crossing_rule` set to `road_crossing` and `segmentation_is_primary` enabled: 200 cities in 84 countries, one hour of footage per city. That run counted 8,915 crossings; 2,514 of them received a reliable road-restricted speed and 482 an initiation time (most pedestrians step onto the road without stopping, or are already on it when first seen). Every run writes its figures to `figures/` as interactive HTML and, when `save_images` is enabled, as PNG and EPS.

![Distribution of crossing speed](docs/images/crossing_speed_histogram.png)
*Crossing speed per pedestrian (median 1.30 m/s).*

![Distribution of crossing initiation time](docs/images/initiation_time_histogram.png)
*Crossing initiation time per pedestrian (median 2.0 s). Waits shorter than three stationary checks (about 1 s, see `check_per_sec_time`) are not recorded.*

![City average crossing speed against initiation time](docs/images/speed_vs_initiation_time.png)
*City averages of crossing speed against initiation time, coloured by continent. Cities with only a few measured waits can show extreme averages (Chișinău), so read single cities with care.*

## Contact
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com or pavlo.bazilinskyy@gmail.com.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
