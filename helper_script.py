# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from typing import Optional, Set
import pandas as pd
from custom_logger import CustomLogger
import common
import subprocess
import logging
from tqdm import tqdm
import pathlib
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logger = CustomLogger(__name__)  # use custom logger
logging.getLogger("ultralytics").setLevel(logging.ERROR)  # Show only errors

# Consts
LINE_THICKNESS = 1
RENDER = False
SHOW_LABELS = False
SHOW_CONF = False

# logging of attempts to upgrade packages
UPGRADE_LOG_FILE = "upgrade_log.json"


class Youtube_Helper:
    """ A helper class for managing YouTube video downloads, processing, and analytics.
    Features:
              - Downloads videos via pytube or yt_dlp with resolution preference.
              - Handles video compression, trimming, and FPS extraction.
              - Applies object detection and tracking using YOLO models.
              - Updates and maintains CSV datasets with video metadata.
              - Interfaces with World Bank data to supplement mapping files.

    Attributes:
        model (str): Path or identifier for the YOLO model to use.
        resolution (str): Target resolution for downloaded videos.
        video_title (str): Title of the currently processed video.
        """

    def __init__(self, video_title=None):
        """
        Initialises a new instance of the class.
        Parameters:
        video_title (str, optional): The title of the video. Defaults to None.
        Instance Variables: self.model (str): The model configuration loaded from common.get_configs("model").
        self.resolution (str): The video resolution. Initialised as None and set later when needed.
        self.video_title (str): The title of the video.
        """
        self.tracking_model = common.get_configs("tracking_model")
        self.segment_model = common.get_configs("segment_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.seg_tracker = common.get_configs("seg_tracker")
        self.resolution = None
        self.mapping = pd.read_csv(common.get_configs("mapping"))
        self.confidence = 0.0
        self.display_frame_tracking = common.get_configs("display_frame_tracking")
        self.display_frame_segmentation = common.get_configs("display_frame_segmentation")
        self.output_path = common.get_configs("videos")
        self.save_annoted_img = common.get_configs("save_annoted_img")
        self.save_tracked_img = common.get_configs("save_tracked_img")
        self.delete_labels = common.get_configs("delete_labels")
        self.delete_frames = common.get_configs("delete_frames")
        self.update_package = common.get_configs("update_package")
        self.need_authentication = common.get_configs("need_authentication")
        self.client = common.get_configs("client")

    def download_videos_from_ftp(self, filename: str, base_url: Optional[str] = None, out_dir: str = ".",
                                 username: Optional[str] = None, password: Optional[str] = None,
                                 token: Optional[str] = None, timeout: int = 20, debug: bool = True,
                                 max_pages: int = 500) -> Optional[tuple[str, str, str, float]]:
        """
        Search and download a specific .mp4 file from a multi-directory FastAPI-based
        HTTP file server (e.g., files.mobility-squad.com). This function attempts direct
        download from known /files/ paths (tue1/tue2/tue3), and if not found, recursively
        crawls the /browse pages to locate the video file. Progress is shown with tqdm.
        Args: filename (str): Target file name (with or without .mp4 extension).
        base_url (str, optional): Base URL of the file server.
        Must include protocol, e.g. "https://files.mobility-squad.com/".
        out_dir (str, optional): Local output directory to save the video.
        Defaults to current directory ".".
        username (str, optional): Username for HTTP Basic Auth.
        password (str, optional): Password for HTTP Basic Auth.
        token (str, optional): Token string for token-based authentication.
        Sent as a query parameter ?token=.... timeout (int, optional):
        Request timeout in seconds. Default is 20. max_pages (int, optional):
        Safety limit for crawl depth/pages. Default is 500.
        Returns: Optional[Tuple[str, str, str, float]]: Returns a tuple
        (local_path, filename, resolution_label, fps) if the download succeeds,
        or None if the file is not found or download fails.
        Logging: - logger.info: start, success summaries.
        - logger.debug: HTTP requests, crawl steps, file matches.
        - logger.warning: non-fatal issues (metadata failures, skipped pages).
        - logger.error: fatal errors (network/IO exceptions). Example:

            result = self.download_videos_from_http_fileserver(
                filename="3ai7SUaPoHM",
                base_url="https://files.mobility-squad.com/",
                out_dir="./downloads",
                username="mobility",
                password="your_password"
            )
            if result:
                path, name, res, fps = result
                print(f"Downloaded {name} ({res}, {fps} fps) to {path}")
            else:
                print("File not found or failed.")
        """
        # -------------------- Input Preparation --------------------
        if not base_url:
            logger.error("Base URL is missing.")
            return None

        base = base_url if base_url.endswith("/") else base_url + "/"

        if username == "":
            username = None
        if password == "":
            password = None

        filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
        filename_lower = filename_with_ext.lower()
        aliases = ["tue1", "tue2", "tue3", "tue4"]

        req_params = {"token": token} if token else None

        logger.info(f"Starting download for '{filename_with_ext}'")
        logger.debug(
            f"Base URL: {base} | Auth: {'Basic' if username and password else 'None'} | Token: {'Yes' if token else 'No'}"  # noqa:E501
        )  # noqa: E501

        # ---------- Session ----------
        with requests.Session() as session:
            if username and password:
                session.auth = (username, password)
            session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

            def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
                """GET with logging and safe error handling."""
                try:
                    r = session.get(url, timeout=timeout, params=req_params, stream=stream)
                    logger.debug(f"GET {url} -> {r.status_code}")
                    if r.status_code == 401:
                        logger.error(f"Authentication failed for {url}")
                    r.raise_for_status()
                    return r
                except requests.RequestException as e:
                    logger.warning(f"Request failed [{url}]: {e}")
                    return None

            # ---------- 1. Try direct /files paths ----------
            for alias in aliases:
                direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
                logger.debug(f"Trying direct URL: {direct_url}")

                r = fetch(direct_url, stream=True)
                if r is None:
                    continue

                logger.info(f"Found file via direct URL: {direct_url}")
                content_len = int(r.headers.get("content-length", 0))
                logger.debug(f"Content-Length: {content_len or 'unknown'} bytes")

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                # Avoid overwriting
                if os.path.exists(local_path):
                    stem, suf = os.path.splitext(local_path)
                    i = 1
                    while os.path.exists(f"{stem} ({i}){suf}"):
                        i += 1
                    local_path = f"{stem} ({i}){suf}"
                    logger.warning(f"File exists, saving as: {local_path}")

                # ---------- Download ----------
                try:
                    total = content_len or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading from ftp: {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Download complete: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download failed for {filename_with_ext}: {e}")
                    return None

                # ---------- Metadata ----------
                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Youtube_Helper.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata extracted: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                logger.info(f"✅ Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
                return local_path, filename, resolution, fps

            # ---------- 2. Crawl /browse fallback ----------
            visited: Set[str] = set()

            def is_dir_link(href: str) -> bool:
                return href.startswith("/v/") and "/browse" in href

            def is_file_link(href: str) -> bool:
                return "/files/" in href

            def crawl(start_url: str) -> Optional[str]:
                """Recursively traverse /browse pages."""
                stack = [start_url]
                pages_seen = 0

                while stack:
                    url = stack.pop()

                    if url in visited:
                        continue

                    visited.add(url)
                    pages_seen += 1
                    if pages_seen > max_pages:
                        logger.warning(f"Crawl aborted after {max_pages} pages.")
                        return None

                    resp = fetch(url)
                    if resp is None:
                        continue

                    try:
                        soup = BeautifulSoup(resp.text, "html.parser")
                    except Exception as e:
                        logger.warning(f"HTML parse failed at {url}: {e}")
                        continue

                    for a in soup.find_all("a"):
                        href = (a.get("href") or "").strip()  # type: ignore
                        if not href:
                            continue

                        full = urljoin(url, href)

                        if is_file_link(href):
                            anchor_text = (a.text or "").strip().lower()
                            tail = pathlib.PurePosixPath(urlparse(full).path).name.lower()
                            if anchor_text == filename_lower or tail == filename_lower:
                                logger.info(f"File located via crawl: {full}")
                                return full

                        if is_dir_link(href):
                            stack.append(full)

                logger.debug("Crawl finished — no file found.")
                return None

            for alias in aliases:
                start_url = urljoin(base, f"v/{alias}/browse")
                logger.debug(f"Crawling alias: {alias} -> {start_url}")

                found = crawl(start_url)
                if not found:
                    continue

                r = fetch(found, stream=True)
                if not r:
                    continue

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                try:
                    total = int(r.headers.get("content-length", 0)) or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Downloaded via crawl: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download during crawl failed: {e}")
                    return None

                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Youtube_Helper.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename, resolution, fps

            logger.warning(f"File '{filename_with_ext}' not found in any alias.")
            return None

    def get_video_fps(self, video_file_path):
        """
        Retrieves the frames per second (FPS) of a video file using OpenCV.
        Parameters: video_file_path (str): The file path to the video whose FPS is to be determined.
        Returns: int or None: The rounded FPS value of the video if successful;
        otherwise, returns None if an error occurs.
        The function performs the following steps:
        1. Opens the video file using OpenCV's VideoCapture.
        2. Retrieves the FPS using the CAP_PROP_FPS property.
        3. Rounds the FPS value to the nearest integer.
        4. Releases the video resource.
        5. Returns the rounded FPS value, or None if an exception is encountered.
        """
        try:
            # Open the video file using OpenCV
            video = cv2.VideoCapture(video_file_path)

            # Retrieve the FPS using OpenCV's CAP_PROP_FPS property
            fps = video.get(cv2.CAP_PROP_FPS)

            # Release the video resource
            video.release()

            # Return the FPS rounded to the nearest integer
            return round(fps, 0)
        except Exception as e:
            # Log an error message if FPS retrieval fails
            logger.error(f"Failed to retrieve FPS: {e}")
            return None

    @staticmethod
    def get_video_resolution_label(video_path: str) -> str:
        """
        Return a resolution label for a local video file using an "exact, truthful" policy.

        Policy
        -----------------
        - Read the frame height (pixels) from the file via OpenCV.
        - If the height matches a known standard, return its label (e.g., "720p", "1080p").
        - If the height is close to a known standard within a small tolerance (to account for
          encoder/container padding such as 1088 instead of 1080), return the nearest
          standard label.
        - Otherwise, return the exact height in the form "<height>p" (e.g., "540p", "768p").

        This approach remains compatible with the updated download selection logic, which
        may select non-standard heights when they are the best available option.

        Parameters
        ----------
        video_path : str
            Path to the video file on disk.

        Returns
        -------
        str
            A resolution label (e.g., "144p", "360p", "720p", "1080p") or "<height>p" for
            non-standard heights.

        Raises
        ------
        FileNotFoundError
            If `video_path` does not exist.
        RuntimeError
            If the video cannot be opened or the frame height cannot be determined.

        Notes
        -----
        - The label is derived from frame height only (not bitrate, codec, aspect ratio, etc.).
        - Some videos may report padded heights (e.g., 544, 736, 1088). These are mapped to
          the nearest standard label only when within the configured tolerance.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        if height <= 0:
            raise RuntimeError(f"Could not determine frame height for video: {video_path}")

        # Canonical/common heights (add more if we want "named" labels, but non-matches still fall back to "<height>p")
        labels = {
            144: "144p",
            240: "240p",
            360: "360p",
            480: "480p",
            540: "540p",
            576: "576p",
            720: "720p",
            900: "900p",
            1080: "1080p",
            1440: "1440p",
            2160: "2160p",  # 4K UHD
            4320: "4320p",  # 8K UHD
        }

        # Small tolerance to normalize padded encodes (e.g., 1088 -> 1080).
        tolerance_px = 16

        # Exact match
        if height in labels:
            return labels[height]

        # Nearest label within tolerance (padding normalization only)
        closest_h = min(labels.keys(), key=lambda h: abs(height - h))
        if abs(height - closest_h) <= tolerance_px:
            return labels[closest_h]

        # Truthful fallback for truly non-standard heights
        return f"{height}p"

    def trim_video(self, input_path, output_path, start_time, end_time):
        """
        Trims a segment from a video and saves the result to a specified file.
        Parameters: input_path (str): The file path to the original video.
        output_path (str): The destination file path where the trimmed video will be saved.
        start_time (float or str): The start time for the trimmed segment.
        This can be specified in seconds or in a time format recognised by MoviePy.
        nd_time (float or str): The end time for the trimmed segment.
        Similar to start_time, it can be in seconds or another supported time format.
        Returns: None The function performs the following steps:
        1. Loads the original video using MoviePy's VideoFileClip.
        2. Creates a subclip from the original video based on the provided start_time and end_time.
        3. Writes the subclip to the output_path using the H.264 video codec and AAC audio codec.
        4. Closes the video file to free up resources.
        """
        # Load the video and create a subclip using the provided start and end times.
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the subclip to the specified output file using the 'libx264' codec for video and 'aac' for audio.
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close the video clip to release any resources used.
        video_clip.close()

    def draw_yolo_boxes_on_video(self, df, fps, video_path, output_path):
        """
        Draw YOLO-style bounding boxes on a video and save the annotated output.
        This method takes a DataFrame containing normalised bounding box coordinates
        (in YOLO format), matches them frame-by-frame to the input video,
        draws the corresponding boxes and labels, and writes the resulting video to disk.
        Args: df (pd.DataFrame): DataFrame containing at least the following columns:
        - 'frame-count': Original frame indices in the source video.
        - 'X-center', 'Y-center': Normalised center coordinates (0 to 1).
        - 'width', 'Height': Normalised width and height (0 to 1).
        - 'unique-id': Identifier to display in the label.
        fps (float): Frames per second for the output video.
        video_path (str): Path to the input video file.
        output_path (str): Path to save the annotated output video.

        Raises: IOError: If the input video cannot be opened.
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Normalise frame indices to start from 0
        min_frame = df["frame-count"].min()
        df["Frame Index"] = df["frame-count"] - min_frame

        # Attempt to open the input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Get video dimensions and total number of frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set up video writer with the same resolution and specified fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Writing to {output_path} ({width}x{height} @ {fps}fps)")

        frame_index = 0

        # Process each frame
        while frame_index < total_frames:
            success, frame = cap.read()
            if not success:
                logger.error(f"Failed to read frame {frame_index}")
                break

            # Filter YOLO data for this adjusted frame index
            frame_data = df[df["Frame Index"] == frame_index]
            for _, row in frame_data.iterrows():
                # Convert normalised coordinates to absolute pixel values
                x_center = row["x-center"] * width
                y_center = row["y-center"] * height
                w = row["width"] * width
                h = row["height"] * height

                # Calculate top-left and bottom-right corners of the box
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Draw rectangle and label with unique ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID: {int(row['unique-id'])}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the modified frame to the output video
            out.write(frame)
            frame_index += 1

        # Release video objects
        cap.release()
        out.release()

    def detect_gpu(self):
        """
        Detects whether an NVIDIA or Intel GPU is available and returns the appropriate FFmpeg encoder.
        Returns: str: 'hevc_nvenc' for NVIDIA, 'hevc_qsv' for Intel, or None if no compatible GPU is found.
        """
        try:
            # Check for NVIDIA GPU
            nvidia_check = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if nvidia_check.returncode == 0:
                return "hevc_nvenc"

            # Check for Intel QuickSync GPU
            intel_check = subprocess.run(["vainfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "Intel" in intel_check.stdout:
                return "hevc_qsv"

        except FileNotFoundError:
            pass  # Command not found, meaning the hardware isn't available or the driver isn't installed

        return None  # No compatible GPU found
