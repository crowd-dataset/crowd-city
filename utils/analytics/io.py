"""Detection-file discovery helpers for CROWD analysis."""

import os
from typing import List, Optional

import polars as pl

import common
from utils.core.metadata import MetaData


metadata_class = MetaData()

SUPPORTED_DETECTION_EXTENSIONS = {".csv", ".parquet"}


class IO:
    def __init__(self) -> None:
        pass


    @staticmethod
    def parquet_detection_files(folder_path: str) -> List[str]:
        """Return visible Parquet detection files from one mirror folder."""
        return [
            name
            for name in os.listdir(folder_path)
            if not name.startswith(".")
            and os.path.splitext(name)[1].lower() == ".parquet"
        ]

    @staticmethod
    def preferred_detection_files(folder_path: str) -> List[str]:
        """Return one detection file per stem, preferring Parquet over CSV.

        The output preserves the directory listing order as closely as
        possible. If both ``sample.csv`` and ``sample.parquet`` exist, only the
        Parquet file is returned so the same video segment is never analysed
        twice.
        """
        entries = os.listdir(folder_path)
        parquet_stems = {
            os.path.splitext(name)[0]
            for name in entries
            if not name.startswith(".")
            and os.path.splitext(name)[1].lower() == ".parquet"
        }

        selected: List[str] = []
        for name in entries:
            if name.startswith("."):
                continue

            stem, extension = os.path.splitext(name)
            extension = extension.lower()
            if extension not in SUPPORTED_DETECTION_EXTENSIONS:
                continue

            if extension == ".csv" and stem in parquet_stems:
                continue

            selected.append(name)

        return selected

    def filter_detection_file(
        self,
        file: str,
        df_mapping: pl.DataFrame,
    ) -> Optional[str]:
        """Validate one CSV or Parquet detection file against the mapping.

        Parquet and CSV use the same filename stem, so the existing metadata
        lookup and vehicle filters remain unchanged.
        """
        original_file = os.fspath(file)
        if original_file.startswith("."):
            return None

        filename, extension = os.path.splitext(original_file)
        if extension.lower() not in SUPPORTED_DETECTION_EXTENSIONS:
            return None

        values = metadata_class.find_values_with_video_id(df_mapping, filename)
        if values is None:
            return None

        vehicle_type = values[18]
        vehicle_list = common.get_configs("vehicles_analyse")
        if vehicle_list and vehicle_type not in vehicle_list:
            return None

        return original_file

    def filter_csv_files(
        self,
        file: str,
        df_mapping: pl.DataFrame,
    ) -> Optional[str]:
        """Backward-compatible alias for ``filter_detection_file``.

        Existing callers can keep using this method while CSV and Parquet are
        both supported.
        """
        return self.filter_detection_file(file=file, df_mapping=df_mapping)
