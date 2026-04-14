## CROWD-city

## Citation and usage of code
If you use this work for academic work please cite the following paper:

> Alam, M. S., Martens, M. H., & Bazilinskyy, P. (2025). 

The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code. For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

## Getting started
[![Python Version](https://img.shields.io/badge/python-3.10.18-blue.svg)](https://www.python.org/downloads/release/python-3919/)
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
The repo should contain a .python-version file so `uv` will automatically use this version.

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

**Step 8:** Ensure that dataset are present. Place required datasets (including **mapping.csv**) into the **data/** directory:


**Step 9:** Run the code:
```command line
python3 analysis.py
```

### Configuration of project
Configuration of the project needs to be defined in `config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
- **`data`**: Directory containing data (CSV output from YOLO).
- **`videos`**: Directories containing the videos used to generate the data.
- **`mapping`**: CSV file that contains mapping data for the cities referenced in the data.
- **`prediction_mode`**: Configures YOLO for object detection.
- **`tracking_mode`**: Configures YOLO for object tracking.
- **`always_analyse`**: Always conduct analysis even when pickle files are present (good for testing).
- **`display_frame_tracking`**: Displays the frame tracking during analysis.
- **`save_annotated_img`**: Saves the annotated frames produced by YOLO.
- **`delete_labels`**: Deletes label files from YOLO output.
- **`delete_frames`**: Deletes frames from YOLO output.
- **`delete_youtube_video`**: Deletes saved YouTube videos.
- **`compress_youtube_video`**: Compresses YouTube videos (using the H.255 codec by default).
- **`delete_runs_files`**: Deletes files containing YOLO output after analysis.
- **`check_missing_mapping`**: Identifies all the missing csv files.
- **`min_max_videos`**: Gives snippets of the fastest and slowest crossing pedestrian.
- **`track_buffer_sec`**: Keep tracks longer (in seconds).
- **`analysis_level`**: Specifies the analysis level; supported versions include `city` and `country`.
- **`client`**: Specifies the client type for downloading YouTube videos; accepted values are `"WEB"`, `"ANDROID"` or `"ios"`.
- **`model`**: Specifies the YOLO model to use; supported/tested versions include `v8x` and `v11x`.
- **`boundary_left`**: Specifies the x-coordinate of one edge of the crossing area used to detect road crossings (normalised between 0 and 1).
- **`boundary_right`**: Specifies the x-coordinate of the opposite edge of the crossing area used to detect road crossings (normalised between 0 and 1).
- **`use_geometry_correction`**: Specifies the distance threshold for applying geometry correction. If set to 0, geometry correction is skipped.
- **`population_threshold`**: Specifies the minimum population a city must have to be included in the analysis.
- **`footage_threshold`**: Specifies the minimum amount of footage required for a city to be included in the analysis.
- **`min_city_population_percentage`**: Specifies the minimum proportion of a country’s population that a city must have to be included in the analysis.
- **`min_speed`**: Specifies the minimum speed limit for pedestrian crossings to be included in the analysis.
- **`max_speed`**: Specifies the maximum speed limit for pedestrian crossings to be included in the analysis.
- **`countries_analyse`**: Lists the countries to be analysed.
- **`confidence`**: Sets the confidence threshold parameter for YOLO.
- **`update_ISO_code`**: Updates the ISO code of the country in the mapping file during analysis.
- **`update_pop_country`**: Updates the country’s population in the mapping file during analysis.
- **`update_gini_value`**: Updates the GINI value of the country in the mapping file during analysis.
- **`update_pytubefix`**: Updates the `pytubefix` library each time analysis starts.
- **`font_family`**: Specifies the font family to be used in outputs.
- **`font_size`**: Specifies the font size to be used in outputs.
- **`plotly_template`**: Defines the template for Plotly figures.
- **`logger_level`**: Level of console output. Can be: debug, info, warning, error.
- **`sleep_sec`**: Amount of seconds of pause in the end of the loop in `main.py`.
- **`git_pull`**: Pull changes from git repository in the end of the loop in `main.py`.
- **`email_send`**: Send email about completion of the job in the end of the loop in `main.py`. See the following paragraph for the additional parameters in the `secret` file.
- **`email_sender`**: Email address of the the "sender" of the email.
- **`email_recipients`**: List of emails for sending the message.
- **`max_workers`**: Specifies the maximum number of segment-processing worker threads (i.e., how many segments can be analysed in parallel). Increasing this increases concurrent segment processing, subject to GPU/CPU and I/O limits.
- **`download_workers`**: Specifies the maximum number of concurrent video download/prepare workers. Increasing this allows multiple videos to be downloaded/prepared in parallel (useful when network/FTP is the bottleneck).
- **`max_active_segments_per_video`**: Specifies the maximum number of segments from the *same video* that are allowed to be processed concurrently.
  - If set to **1**, the scheduler tends to distribute workers across **different videos** (e.g., with `max_workers=3`, it will try to process 3 different videos at once).
  - If set to **2+**, multiple workers may process segments from the **same video** simultaneously, which can improve throughput when one video has many segments but reduces “video diversity” across workers.


For working with external APIs of [VideoFiles](https://files.mobility-squad.com/), [GeoNames](https://www.geonames.org), [BEA](https://apps.bea.gov/api/signup), [TomTom](https://developer.tomtom.com/user/register), [Trafikab](https://www.trafiklab.se/api/trafiklab-apis), and [Numbeo](https://www.numbeo.com/common/api.jsp) (paid), the API keys need to be placed in file `secret` (no extension) in the root of the project. The file needs to be formatted as `default.secret`. The email SMTP server, account and password need to be also set here. This is optional for just running the analysis on the dataset. For running the the `main.py` script at least an empty `secret` file directly copies from the template is required.


## Contact
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com or pavlo.bazilinskyy@gmail.com.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
