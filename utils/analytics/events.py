import os
from typing import Dict, List, Optional, Tuple

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from utils.core.metadata import MetaData

metadata_class = MetaData()
logger = CustomLogger(__name__)


class Events:
    """Analyse crossing events using detection data from the Parquet store."""

    _detection_index: Dict[str, str] = {}
    _detection_index_signature: Tuple[str, ...] = ()

    def __init__(self) -> None:
        pass

    @staticmethod
    def _find_detection_path(video_key: str) -> Optional[str]:
        """Return the first configured Parquet path for one video key."""
        for root in common.get_configs("parquet_data") or []:
            file_path = os.path.join(
                os.fspath(root),
                "bbox",
                f"{video_key}.parquet",
            )
            if os.path.isfile(file_path):
                return file_path
        return None

    @classmethod
    def _load_detection_data(
        cls,
        video_key: str,
    ) -> Optional[pl.DataFrame]:
        """
        Load one detection file from the persistent Parquet store.

        Only columns required by the event calculations are materialised.
        The confidence filter is applied lazily before collection.

        Detection schemas are normalised here because historical Parquet files
        may store the same field with different inferred types. Tracker IDs are
        treated as opaque strings, while frame and YOLO class IDs are numeric.
        """
        file_path = cls._find_detection_path(str(video_key))
        if file_path is None:
            return None

        min_conf = float(common.get_configs("min_confidence"))

        try:
            lazy = pl.scan_parquet(file_path)
            schema_names = set(lazy.collect_schema().names())

            required = {"unique-id", "frame-count", "yolo-id"}
            if not required.issubset(schema_names):
                logger.warning(
                    f"{file_path}: required event columns are missing."
                )
                return None

            expressions = [
                pl.col("unique-id")
                .cast(pl.Utf8, strict=False)
                .alias("unique-id"),
                pl.col("frame-count")
                .cast(pl.Int64, strict=False)
                .alias("frame-count"),
                pl.col("yolo-id")
                .cast(pl.Int64, strict=False)
                .alias("yolo-id"),
            ]

            if "confidence" in schema_names:
                lazy = lazy.filter(
                    pl.col("confidence").cast(
                        pl.Float64,
                        strict=False,
                    ) >= min_conf
                )
                expressions.append(
                    pl.col("confidence")
                    .cast(pl.Float64, strict=False)
                    .alias("confidence")
                )

            return lazy.select(expressions).collect()

        except Exception as exc:
            logger.error(
                f"Could not load Parquet detection data for "
                f"{video_key}: {exc}"
            )
            return None

    @staticmethod
    def _normalise_crossing_ids(values) -> List[str]:
        """Return crossing tracker IDs in the same string format as event data."""
        output: List[str] = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if not text or text.lower() in {"nan", "none", "null"}:
                continue
            try:
                numeric = float(text)
                if numeric.is_integer():
                    text = str(int(numeric))
            except (TypeError, ValueError):
                pass
            output.append(text)
        return output

    @staticmethod
    def crossing_event_with_traffic_equipment(
        df_mapping: pl.DataFrame,
        data: dict,
    ):
        """
        Analyse pedestrian crossing events in relation to traffic equipment.

        Traffic equipment is represented by YOLO IDs 9 and 11. Counts are
        aggregated by locality/condition and country/condition.
        """
        total_duration_by_locality = {}
        total_duration_by_country = {}
        crossings_with_traffic_equipment_locality = {}
        crossings_with_traffic_equipment_country = {}
        crossings_without_traffic_equipment_locality = {}
        crossings_without_traffic_equipment_country = {}

        for video_key, crossings in tqdm(
            data.items(),
            total=len(data),
        ):
            result = metadata_class.find_values_with_video_id(
                df_mapping,
                video_key,
            )
            if result is None:
                continue

            start_time = result[1]
            end_time = result[2]
            condition = result[3]
            locality = result[4]
            latitude = result[6]
            longitude = result[7]
            country = result[8]

            location_key_locality = (
                f"{locality}_{latitude}_{longitude}_{condition}"
            )
            location_key_country = f"{country}_{condition}"

            try:
                duration = end_time - start_time
            except Exception:
                continue

            total_duration_by_locality[location_key_locality] = (
                total_duration_by_locality.get(
                    location_key_locality,
                    0,
                )
                + duration
            )
            total_duration_by_country[location_key_country] = (
                total_duration_by_country.get(
                    location_key_country,
                    0,
                )
                + duration
            )

            value = Events._load_detection_data(video_key)
            if value is None:
                continue

            required = {"unique-id", "frame-count", "yolo-id"}
            if not required.issubset(set(value.columns)):
                continue

            count_with_equipment = 0
            count_without_equipment = 0

            uids = Events._normalise_crossing_ids(crossings.keys())
            if not uids:
                continue

            ranges = (
                value
                .filter(pl.col("unique-id").is_in(uids))
                .group_by("unique-id")
                .agg([
                    pl.col("frame-count")
                    .min()
                    .alias("_fmin"),
                    pl.col("frame-count")
                    .max()
                    .alias("_fmax"),
                ])
            )

            for _uid, fmin, fmax in ranges.iter_rows():
                if fmin is None or fmax is None:
                    continue

                seg = value.filter(
                    (pl.col("frame-count") >= int(fmin))
                    & (pl.col("frame-count") <= int(fmax))
                )

                has_equipment = seg.select(
                    pl.col("yolo-id").is_in([9, 11]).any()
                ).item()

                if bool(has_equipment):
                    count_with_equipment += 1
                else:
                    count_without_equipment += 1

            crossings_with_traffic_equipment_locality[
                location_key_locality
            ] = (
                crossings_with_traffic_equipment_locality.get(
                    location_key_locality,
                    0,
                )
                + count_with_equipment
            )
            crossings_without_traffic_equipment_locality[
                location_key_locality
            ] = (
                crossings_without_traffic_equipment_locality.get(
                    location_key_locality,
                    0,
                )
                + count_without_equipment
            )

            crossings_with_traffic_equipment_country[
                location_key_country
            ] = (
                crossings_with_traffic_equipment_country.get(
                    location_key_country,
                    0,
                )
                + count_with_equipment
            )
            crossings_without_traffic_equipment_country[
                location_key_country
            ] = (
                crossings_without_traffic_equipment_country.get(
                    location_key_country,
                    0,
                )
                + count_without_equipment
            )

        return (
            crossings_with_traffic_equipment_locality,
            crossings_without_traffic_equipment_locality,
            total_duration_by_locality,
            crossings_with_traffic_equipment_country,
            crossings_without_traffic_equipment_country,
            total_duration_by_country,
        )

    @staticmethod
    def crossing_event_wt_traffic_light(
        df_mapping: pl.DataFrame,
        data: dict,
    ):
        """
        Calculate the percentage of crossing events without a traffic light.

        Detection data is loaded from the persistent Parquet store.
        """
        var_exist = {}
        var_nt_exist = {}
        ratio = {}
        time_ = []

        counter_1 = {}
        counter_2 = {}

        for key, df_ids in tqdm(
            data.items(),
            total=len(data),
        ):
            counter_exists = 0
            counter_nt_exists = 0

            result = metadata_class.find_values_with_video_id(
                df_mapping,
                key,
            )
            if result is None:
                continue

            start = result[1]
            end = result[2]
            condition = result[3]
            locality = result[4]
            lat = result[6]
            long = result[7]

            duration = end - start
            time_.append(duration)

            value = Events._load_detection_data(key)
            if value is None:
                continue

            required = {"unique-id", "frame-count", "yolo-id"}
            if not required.issubset(set(value.columns)):
                continue

            ids = Events._normalise_crossing_ids(df_ids.keys())
            if not ids:
                continue

            ranges = (
                value
                .filter(pl.col("unique-id").is_in(ids))
                .group_by("unique-id")
                .agg([
                    pl.col("frame-count")
                    .min()
                    .alias("_fmin"),
                    pl.col("frame-count")
                    .max()
                    .alias("_fmax"),
                ])
            )

            for _uid, fmin, fmax in ranges.iter_rows():
                if fmin is None or fmax is None:
                    continue

                seg = value.filter(
                    (pl.col("frame-count") >= int(fmin))
                    & (pl.col("frame-count") <= int(fmax))
                )

                yolo_9_exists = seg.select(
                    (pl.col("yolo-id") == 9).any()
                ).item()

                if bool(yolo_9_exists):
                    counter_exists += 1
                else:
                    counter_nt_exists += 1

            if time_[-1] > 0:
                var_exist[key] = (
                    counter_exists * 60
                ) / time_[-1]
                var_nt_exist[key] = (
                    counter_nt_exists * 60
                ) / time_[-1]
            else:
                var_exist[key] = 0
                var_nt_exist[key] = 0

            locality_id_format = (
                f"{locality}_{lat}_{long}_{condition}"
            )

            counter_1[locality_id_format] = (
                counter_1.get(locality_id_format, 0)
                + var_exist[key]
            )
            counter_2[locality_id_format] = (
                counter_2.get(locality_id_format, 0)
                + var_nt_exist[key]
            )

            denom = (
                counter_1[locality_id_format]
                + counter_2[locality_id_format]
            )
            if denom == 0:
                continue

            ratio[locality_id_format] = (
                counter_2[locality_id_format] * 100
            ) / denom

        return ratio
