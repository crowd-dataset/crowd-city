import math
import os
import pickle
import statistics

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import common
from custom_logger import CustomLogger
from utils.core.dataset_stats import Dataset_Stats
from utils.core.iso import ISO
from utils.core.metadata import MetaData
from utils.plotting import constants as C
from utils.plotting.io import IO
from utils.plotting.layout import Layout


layout_class = Layout()
metadata_class = MetaData()
plots_io_class = IO()
iso_class = ISO()
dataset_stats = Dataset_Stats()

logger = CustomLogger(__name__)

file_results = "results.pickle"


def crossing_metric_label(mean: bool = False) -> str:
    metric = (
        "crossing speed (m/s)"
        if os.environ.get("CROWD_CROSSING_SPEED_UNIT") == "m/s"
        else "relative crossing motion index"
    )
    return f"Mean {metric}" if mean else metric.capitalize()


class Crossings:
    """
    Plotting utilities for CROWD crossing metrics.

    The results.pickle tuple layout used by analysis.py is:

        24  avg_time_locality
        25  avg_speed_locality
        27  avg_speed_country
        28  avg_time_country
        29  crossings_with_traffic_equipment_locality
        30  crossings_without_traffic_equipment_locality
        35  pedestrian_cross_country

    Locality keys use:
        {locality}_{latitude}_{longitude}_{condition}

    Parsing is always performed from the right so locality names containing
    underscores remain valid.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_results():
        with open(file_results, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _parse_locality_condition_key(key: str):
        """
        Parse:
            {locality}_{latitude}_{longitude}_{condition}

        rsplit is intentional because locality itself may contain underscores.
        """
        parts = str(key).rsplit("_", 3)
        if len(parts) != 4:
            raise ValueError(
                "Invalid locality condition key "
                f"{key!r}; expected locality_latitude_longitude_condition."
            )

        locality, lat, lon, condition = parts

        try:
            lat_float = float(lat)
            lon_float = float(lon)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid coordinates in locality condition key {key!r}."
            ) from exc

        return locality, lat, lon, lat_float, lon_float, str(condition)

    @staticmethod
    def _parse_locality_key(key: str):
        """
        Parse:
            {locality}_{latitude}_{longitude}
        """
        parts = str(key).rsplit("_", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid locality key {key!r}; expected locality_latitude_longitude."
            )

        locality, lat, lon = parts

        try:
            lat_float = float(lat)
            lon_float = float(lon)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid coordinates in locality key {key!r}."
            ) from exc

        return locality, lat, lon, lat_float, lon_float

    @staticmethod
    def _parse_country_condition_key(key: str):
        parts = str(key).rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid country condition key {key!r}; expected country_condition."
            )
        return parts[0], parts[1]

    @staticmethod
    def _as_float(value):
        if value is None:
            return None

        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value_float):
            return None

        return value_float

    @staticmethod
    def _safe_average(values):
        valid = [
            float(value)
            for value in values
            if Crossings._as_float(value) is not None
        ]

        if not valid:
            return None

        return sum(valid) / len(valid)

    @staticmethod
    def _is_missing_text(value) -> bool:
        if value is None:
            return True

        if isinstance(value, float) and math.isnan(value):
            return True

        if isinstance(value, str):
            return value.strip().lower() in {"", "nan", "na", "n/a", "unknown"}

        return False

    @staticmethod
    def _flag_from_iso3(iso3):
        if Crossings._is_missing_text(iso3):
            return ""

        try:
            iso2 = iso_class.iso3_to_iso2(str(iso3))
            if not iso2:
                return ""
            return iso_class.iso2_to_flag(iso2) or ""
        except Exception:
            return ""

    @staticmethod
    def _locality_metadata(df_mapping, locality: str, lat: float):
        country = metadata_class.get_value(
            df_mapping,
            "locality",
            locality,
            "lat",
            lat,
            "country",
        )
        iso3 = metadata_class.get_value(
            df_mapping,
            "locality",
            locality,
            "lat",
            lat,
            "iso3",
        )
        state = metadata_class.get_value(
            df_mapping,
            "locality",
            locality,
            "lat",
            lat,
            "state",
        )
        return country, iso3, state

    @staticmethod
    def _format_locality_label(
        df_mapping,
        locality: str,
        lat: float,
        iso3=None,
        state=None,
    ) -> str:
        if state is None:
            state = metadata_class.get_value(
                df_mapping,
                "locality",
                locality,
                "lat",
                lat,
                "state",
            )

        if iso3 is None:
            iso3 = metadata_class.get_value(
                df_mapping,
                "locality",
                locality,
                "lat",
                lat,
                "iso3",
            )

        if Crossings._is_missing_text(state):
            label = str(locality)
        else:
            label = f"{locality}, {state}"

        flag = Crossings._flag_from_iso3(iso3)
        return f"{flag} {label}".strip()

    @staticmethod
    def _format_country_label(df_mapping, country: str, iso3=None) -> str:
        if iso3 is None:
            iso3 = metadata_class.get_value(
                df=df_mapping,
                column_name1="country",
                column_value1=country,
                column_name2=None,
                column_value2=None,
                target_column="iso3",
            )

        flag = Crossings._flag_from_iso3(iso3)
        return f"{flag} {country}".strip()

    @staticmethod
    def _log_values(name: str, day_values, night_values) -> None:
        day_values = [
            float(value)
            for value in day_values
            if Crossings._as_float(value) is not None
        ]
        night_values = [
            float(value)
            for value in night_values
            if Crossings._as_float(value) is not None
        ]

        if day_values:
            logger.info(
                f"Mean {name} during daytime: {statistics.mean(day_values):.2f}"
            )
            logger.info(
                f"Standard deviation of {name} during daytime: "
                f"{statistics.stdev(day_values) if len(day_values) > 1 else 0:.2f}"
            )
        else:
            logger.info(f"No valid {name} values found for daytime.")

        if night_values:
            logger.info(
                f"Mean {name} during night time: {statistics.mean(night_values):.2f}"
            )
            logger.info(
                f"Standard deviation of {name} during night time: "
                f"{statistics.stdev(night_values) if len(night_values) > 1 else 0:.2f}"
            )
        else:
            logger.info(f"No valid {name} values found for night time.")

    @staticmethod
    def _log_day_night_differences(records, metric_prefix: str, label_name: str) -> None:
        differences = []

        for record_key, record in records.items():
            day = Crossings._as_float(record.get(f"{metric_prefix}_0"))
            night = Crossings._as_float(record.get(f"{metric_prefix}_1"))

            if day is None or night is None:
                continue

            differences.append((record_key, abs(day - night)))

        if not differences:
            logger.info(
                f"No valid paired day and night {label_name} values found for comparison."
            )
            return

        ordered = sorted(differences, key=lambda item: item[1], reverse=True)
        top = ordered[:5]
        bottom = ordered[-5:]

        logger.info(
            f"Top 5 locations with largest absolute day and night "
            f"{label_name} differences:"
        )
        for location, difference in top:
            logger.info(f"{location}: {difference:.2f}")

        logger.info(
            f"Top 5 locations with smallest absolute day and night "
            f"{label_name} differences:"
        )
        for location, difference in bottom:
            logger.info(f"{location}: {difference:.2f}")

    @staticmethod
    def _nonzero_max(values, default: float = 1.0) -> float:
        valid = [
            float(value)
            for value in values
            if Crossings._as_float(value) is not None and float(value) > 0
        ]

        if not valid:
            return default

        return max(valid)

    # ------------------------------------------------------------------
    # Locality speed and crossing initiation time
    # ------------------------------------------------------------------

    def _build_locality_speed_time_records(self, df_mapping, avg_speed, avg_time):
        records = {}

        if not isinstance(avg_speed, dict) or not isinstance(avg_time, dict):
            return records

        common_keys = set(avg_speed).intersection(avg_time)

        for key in common_keys:
            try:
                locality, lat, lon, lat_float, _, condition = (
                    self._parse_locality_condition_key(key)
                )
            except ValueError as exc:
                logger.warning(str(exc))
                continue

            speed = self._as_float(avg_speed.get(key))
            time_value = self._as_float(avg_time.get(key))

            if speed is None and time_value is None:
                continue

            country, iso3, state = self._locality_metadata(
                df_mapping,
                locality,
                lat_float,
            )

            base_key = f"{locality}_{lat}_{lon}"

            if base_key not in records:
                records[base_key] = {
                    "locality": locality,
                    "lat": lat_float,
                    "lon": float(lon),
                    "country": country,
                    "iso3": iso3,
                    "state": state,
                    "speed_0": None,
                    "speed_1": None,
                    "time_0": None,
                    "time_1": None,
                }

            if condition in {"0", "1"}:
                records[base_key][f"speed_{condition}"] = speed
                records[base_key][f"time_{condition}"] = time_value

        return records

    def speed_and_time_to_start_cross(
        self,
        df_mapping,
        font_size_captions=40,
        x_axis_title_height=150,
        legend_x=0.81,
        legend_y=0.98,
        legend_spacing=0.02,
    ):
        logger.info("Plotting speed_and_time_to_start_cross")

        data_tuple = self._load_results()

        avg_time = data_tuple[24]
        avg_speed = data_tuple[25]

        records = self._build_locality_speed_time_records(
            df_mapping,
            avg_speed,
            avg_time,
        )

        if not records:
            logger.warning(
                "No locality speed and time records are available for plotting."
            )
            return

        self._log_day_night_differences(
            records,
            "speed",
            "crossing motion",
        )
        self._log_day_night_differences(
            records,
            "time",
            "crossing initiation time",
        )

        self._log_values(
            "relative crossing motion",
            [record["speed_0"] for record in records.values()],
            [record["speed_1"] for record in records.values()],
        )
        self._log_values(
            "crossing initiation time",
            [record["time_0"] for record in records.values()],
            [record["time_1"] for record in records.values()],
        )

        ordered_keys = sorted(
            records,
            key=lambda key: (
                str(records[key].get("iso3") or ""),
                str(records[key].get("locality") or ""),
                float(records[key].get("lat") or 0),
            ),
        )

        self._plot_speed_time_records(
            df_mapping=df_mapping,
            records=records,
            ordered_keys=ordered_keys,
            locality_mode=True,
            font_size_captions=font_size_captions,
            x_axis_title_height=x_axis_title_height,
            legend_x=legend_x,
            legend_y=legend_y,
            legend_spacing=legend_spacing,
        )

    # ------------------------------------------------------------------
    # Country speed and crossing initiation time
    # ------------------------------------------------------------------

    def _build_country_speed_time_records(
        self,
        df_mapping,
        avg_speed,
        avg_time,
        no_of_crossing,
    ):
        records = {}

        if not isinstance(avg_speed, dict) or not isinstance(avg_time, dict):
            return records

        no_of_crossing = no_of_crossing if isinstance(no_of_crossing, dict) else {}
        threshold = float(common.get_configs("min_crossing_detect"))

        common_keys = set(avg_speed).intersection(avg_time)

        for key in common_keys:
            crossing_count = self._as_float(no_of_crossing.get(key, 0)) or 0.0

            if crossing_count < threshold:
                continue

            try:
                country, condition = self._parse_country_condition_key(key)
            except ValueError as exc:
                logger.warning(str(exc))
                continue

            if condition not in {"0", "1"}:
                continue

            speed = self._as_float(avg_speed.get(key))
            time_value = self._as_float(avg_time.get(key))

            iso3 = metadata_class.get_value(
                df=df_mapping,
                column_name1="country",
                column_value1=country,
                column_name2=None,
                column_value2=None,
                target_column="iso3",
            )

            if country not in records:
                records[country] = {
                    "country": country,
                    "iso3": iso3,
                    "speed_0": None,
                    "speed_1": None,
                    "time_0": None,
                    "time_1": None,
                }

            records[country][f"speed_{condition}"] = speed
            records[country][f"time_{condition}"] = time_value

        return records

    def speed_and_time_to_start_cross_country(
        self,
        df_mapping,
        font_size_captions=40,
        x_axis_title_height=150,
        legend_x=0.81,
        legend_y=0.98,
        legend_spacing=0.02,
    ):
        logger.info("Plotting speed_and_time_to_start_cross_country")

        data_tuple = self._load_results()

        avg_speed = data_tuple[27]
        avg_time = data_tuple[28]
        no_of_crossing = data_tuple[35]

        records = self._build_country_speed_time_records(
            df_mapping,
            avg_speed,
            avg_time,
            no_of_crossing,
        )

        if not records:
            logger.warning(
                "No country speed and time records are available for plotting."
            )
            return

        self._log_day_night_differences(
            records,
            "speed",
            "crossing motion",
        )
        self._log_day_night_differences(
            records,
            "time",
            "crossing initiation time",
        )

        self._log_values(
            "relative crossing motion",
            [record["speed_0"] for record in records.values()],
            [record["speed_1"] for record in records.values()],
        )
        self._log_values(
            "crossing initiation time",
            [record["time_0"] for record in records.values()],
            [record["time_1"] for record in records.values()],
        )

        ordered_keys = sorted(records)

        self._plot_speed_time_records(
            df_mapping=df_mapping,
            records=records,
            ordered_keys=ordered_keys,
            locality_mode=False,
            font_size_captions=font_size_captions,
            x_axis_title_height=x_axis_title_height,
            legend_x=legend_x,
            legend_y=legend_y,
            legend_spacing=legend_spacing,
        )

    # ------------------------------------------------------------------
    # Shared speed and time figure
    # ------------------------------------------------------------------

    def _plot_speed_time_records(
        self,
        df_mapping,
        records,
        ordered_keys,
        locality_mode,
        font_size_captions,
        x_axis_title_height,
        legend_x,
        legend_y,
        legend_spacing,
    ):
        count = len(ordered_keys)

        if count == 0:
            return

        per_column = (count + 1) // 2
        rows = max(2, per_column * 2)

        fig = make_subplots(
            rows=rows,
            cols=2,
            vertical_spacing=0,
            horizontal_spacing=0.01,
            row_heights=[1.0] * rows,
        )

        speed_totals = []
        time_totals = []

        for record in records.values():
            speed_totals.append(
                (self._as_float(record.get("speed_0")) or 0)
                + (self._as_float(record.get("speed_1")) or 0)
            )
            time_totals.append(
                (self._as_float(record.get("time_0")) or 0)
                + (self._as_float(record.get("time_1")) or 0)
            )

        speed_max = self._nonzero_max(speed_totals) * 1.05
        time_max = self._nonzero_max(time_totals) * 1.05

        for absolute_index, record_key in enumerate(ordered_keys):
            record = records[record_key]

            if absolute_index < per_column:
                column = 1
                local_index = absolute_index
                column_count = min(per_column, count)
            else:
                column = 2
                local_index = absolute_index - per_column
                column_count = count - per_column

            speed_row = local_index * 2 + 1
            time_row = local_index * 2 + 2

            if locality_mode:
                label = self._format_locality_label(
                    df_mapping,
                    record["locality"],
                    record["lat"],
                    iso3=record.get("iso3"),
                    state=record.get("state"),
                )
            else:
                label = self._format_country_label(
                    df_mapping,
                    record["country"],
                    iso3=record.get("iso3"),
                )

            speed_day = self._as_float(record.get("speed_0"))
            speed_night = self._as_float(record.get("speed_1"))
            time_day = self._as_float(record.get("time_0"))
            time_night = self._as_float(record.get("time_1"))

            speed_average = self._safe_average([speed_day, speed_night])
            time_average = self._safe_average([time_day, time_night])

            speed_label = (
                f"{label} {speed_average:.2f}"
                if speed_average is not None
                else label
            )
            time_label = (
                f"{label} {time_average:.2f}"
                if time_average is not None
                else label
            )

            if speed_day is not None:
                fig.add_trace(
                    go.Bar(
                        x=[speed_day],
                        y=[speed_label],
                        orientation="h",
                        name=f"{label} crossing motion during day",
                        marker=dict(color=C.BAR_COLOR_1),
                        showlegend=False,
                        text=[""],
                    ),
                    row=speed_row,
                    col=column,
                )

            if speed_night is not None:
                fig.add_trace(
                    go.Bar(
                        x=[speed_night],
                        y=[speed_label],
                        orientation="h",
                        name=f"{label} crossing motion during night",
                        marker=dict(color=C.BAR_COLOR_2),
                        showlegend=False,
                        text=[""],
                    ),
                    row=speed_row,
                    col=column,
                )

            if time_day is not None:
                fig.add_trace(
                    go.Bar(
                        x=[time_day],
                        y=[time_label],
                        orientation="h",
                        name=f"{label} crossing initiation during day",
                        marker=dict(color=C.BAR_COLOR_3),
                        showlegend=False,
                        text=[""],
                    ),
                    row=time_row,
                    col=column,
                )

            if time_night is not None:
                fig.add_trace(
                    go.Bar(
                        x=[time_night],
                        y=[time_label],
                        orientation="h",
                        name=f"{label} crossing initiation during night",
                        marker=dict(color=C.BAR_COLOR_4),
                        showlegend=False,
                        text=[""],
                    ),
                    row=time_row,
                    col=column,
                )

            fig.update_xaxes(
                range=[0, speed_max],
                side="top",
                showticklabels=(local_index == 0),
                showgrid=True,
                row=speed_row,
                col=column,
            )

            fig.update_xaxes(
                range=[0, time_max],
                side="bottom",
                showticklabels=(local_index == column_count - 1),
                showgrid=True,
                row=time_row,
                col=column,
            )

        for column in (1, 2):
            fig.update_xaxes(
                title=dict(
                    text=crossing_metric_label(mean=True),
                    font=dict(size=font_size_captions),
                ),
                tickfont=dict(size=font_size_captions),
                ticks="outside",
                ticklen=10,
                tickwidth=2,
                tickcolor="black",
                row=1,
                col=column,
            )

        left_bottom_row = min(per_column, count) * 2
        if left_bottom_row >= 1:
            fig.update_xaxes(
                title=dict(
                    text="Mean time to start crossing (in s)",
                    font=dict(size=font_size_captions),
                ),
                tickfont=dict(size=font_size_captions),
                ticks="outside",
                ticklen=10,
                tickwidth=2,
                tickcolor="black",
                row=left_bottom_row,
                col=1,
            )

        right_count = max(0, count - per_column)
        if right_count > 0:
            fig.update_xaxes(
                title=dict(
                    text="Mean time to start crossing (in s)",
                    font=dict(size=font_size_captions),
                ),
                tickfont=dict(size=font_size_captions),
                ticks="outside",
                ticklen=10,
                tickwidth=2,
                tickcolor="black",
                row=right_count * 2,
                col=2,
            )

        fig.update_yaxes(
            showticklabels=True,
            ticklabelposition="inside",
            tickfont=dict(size=14, color="black"),
            showgrid=False,
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            barmode="stack",
            height=max(
                C.BASE_HEIGHT_PER_ROW * per_column * 2,
                C.BASE_HEIGHT_PER_ROW * 2,
            ),
            width=4960,
            showlegend=False,
            margin=dict(
                l=80,
                r=80,
                t=x_axis_title_height,
                b=x_axis_title_height,
            ),
            bargap=0,
            bargroupgap=0,
            font=dict(family=common.get_configs("font_family")),
        )

        legend_items = [
            {
                "name": f"{crossing_metric_label()} during daytime",
                "color": C.BAR_COLOR_1,
            },
            {
                "name": f"{crossing_metric_label()} during night time",
                "color": C.BAR_COLOR_2,
            },
            {
                "name": "Crossing initiation time during daytime",
                "color": C.BAR_COLOR_3,
            },
            {
                "name": "Crossing initiation time during night time",
                "color": C.BAR_COLOR_4,
            },
        ]

        layout_class.add_vertical_legend_annotations(
            fig,
            legend_items,
            x_position=legend_x,
            y_start=legend_y,
            spacing=legend_spacing,
            font_size=font_size_captions,
        )

        plots_io_class.save_plotly_figure(
            fig,
            "consolidated",
            height=max(
                C.BASE_HEIGHT_PER_ROW * per_column * 2,
                C.BASE_HEIGHT_PER_ROW * 2,
            ),
            width=4960,
            scale=C.SCALE,
            save_final=True,
            save_eps=False,
            save_png=False,
        )

    # ------------------------------------------------------------------
    # Traffic equipment figures
    # ------------------------------------------------------------------

    def _build_traffic_equipment_records(
        self,
        df_mapping,
        source,
        value_prefix,
    ):
        records = {}

        if not isinstance(source, dict):
            return records

        for locality_condition, raw_count in source.items():
            try:
                locality, lat, lon, lat_float, _, condition = (
                    self._parse_locality_condition_key(locality_condition)
                )
            except ValueError as exc:
                logger.warning(str(exc))
                continue

            if condition not in {"0", "1"}:
                continue

            count = self._as_float(raw_count)
            if count is None:
                continue

            country, iso3, state = self._locality_metadata(
                df_mapping,
                locality,
                lat_float,
            )

            total_time = self._as_float(
                metadata_class.get_value(
                    df_mapping,
                    "locality",
                    locality,
                    "lat",
                    lat_float,
                    "total_time",
                )
            )
            person = self._as_float(
                metadata_class.get_value(
                    df_mapping,
                    "locality",
                    locality,
                    "lat",
                    lat_float,
                    "person",
                )
            )

            if total_time is None or total_time <= 0:
                logger.warning(
                    f"Skipping {locality_condition}: invalid total_time={total_time}."
                )
                continue

            if person is None or person <= 0:
                logger.warning(
                    f"Skipping {locality_condition}: invalid person count={person}."
                )
                continue

            normalised = count / total_time / person
            normalised = round(normalised * 10**6, 2)

            base_key = f"{locality}_{lat}_{lon}"

            if base_key not in records:
                records[base_key] = {
                    "locality": locality,
                    "lat": lat_float,
                    "lon": float(lon),
                    "country": country,
                    "iso3": iso3,
                    "state": state,
                    f"{value_prefix}_0": None,
                    f"{value_prefix}_1": None,
                }

            records[base_key][f"{value_prefix}_{condition}"] = normalised

        return records

    def _plot_traffic_equipment(
        self,
        df_mapping,
        source,
        value_prefix,
        axis_title,
        output_file,
        font_size_captions,
        x_axis_title_height,
        legend_x,
        legend_y,
        legend_spacing,
    ):
        records = self._build_traffic_equipment_records(
            df_mapping,
            source,
            value_prefix,
        )

        if not records:
            logger.warning(
                f"No data available for {axis_title.lower()}."
            )
            return

        ordered_keys = sorted(
            records,
            key=lambda key: (
                self._safe_average(
                    [
                        records[key].get(f"{value_prefix}_0"),
                        records[key].get(f"{value_prefix}_1"),
                    ]
                )
                or 0
            ),
            reverse=True,
        )

        count = len(ordered_keys)
        per_column = (count + 1) // 2
        rows = max(1, per_column)

        fig = make_subplots(
            rows=rows,
            cols=2,
            vertical_spacing=0.0005,
            horizontal_spacing=0.01,
            row_heights=[1.0] * rows,
        )

        totals = []

        for record in records.values():
            totals.append(
                (self._as_float(record.get(f"{value_prefix}_0")) or 0)
                + (self._as_float(record.get(f"{value_prefix}_1")) or 0)
            )

        x_max = self._nonzero_max(totals) * 1.05

        for absolute_index, record_key in enumerate(ordered_keys):
            record = records[record_key]

            if absolute_index < per_column:
                column = 1
                row = absolute_index + 1
            else:
                column = 2
                row = absolute_index - per_column + 1

            day_value = self._as_float(record.get(f"{value_prefix}_0"))
            night_value = self._as_float(record.get(f"{value_prefix}_1"))

            average = self._safe_average([day_value, night_value])

            label = self._format_locality_label(
                df_mapping,
                record["locality"],
                record["lat"],
                iso3=record.get("iso3"),
                state=record.get("state"),
            )

            if average is not None:
                label = f"{label} {average:.2f}"

            if day_value is not None:
                fig.add_trace(
                    go.Bar(
                        x=[day_value],
                        y=[label],
                        orientation="h",
                        name=f"{label} during day",
                        marker=dict(color=C.BAR_COLOR_1),
                        text=[""],
                        showlegend=False,
                    ),
                    row=row,
                    col=column,
                )

            if night_value is not None:
                fig.add_trace(
                    go.Bar(
                        x=[night_value],
                        y=[label],
                        orientation="h",
                        name=f"{label} during night",
                        marker=dict(color=C.BAR_COLOR_2),
                        text=[""],
                        showlegend=False,
                    ),
                    row=row,
                    col=column,
                )

            fig.update_xaxes(
                range=[0, x_max],
                showgrid=True,
                showticklabels=(row == 1),
                side="top",
                row=row,
                col=column,
            )

        for column in (1, 2):
            fig.update_xaxes(
                title=dict(
                    text=axis_title,
                    font=dict(size=font_size_captions),
                ),
                tickfont=dict(size=font_size_captions),
                ticks="outside",
                ticklen=10,
                tickwidth=2,
                tickcolor="black",
                row=1,
                col=column,
            )

        fig.update_yaxes(
            showticklabels=True,
            ticklabelposition="inside",
            tickfont=dict(size=12, color="black"),
            showgrid=False,
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            barmode="stack",
            height=max(C.BASE_HEIGHT_PER_ROW * rows, C.BASE_HEIGHT_PER_ROW),
            width=2480,
            showlegend=False,
            margin=dict(
                l=80,
                r=100,
                t=x_axis_title_height,
                b=180,
            ),
            bargap=0,
            bargroupgap=0,
            font=dict(family=common.get_configs("font_family")),
        )

        legend_items = [
            {"name": "Day", "color": C.BAR_COLOR_1},
            {"name": "Night", "color": C.BAR_COLOR_2},
        ]

        layout_class.add_vertical_legend_annotations(
            fig,
            legend_items,
            x_position=legend_x,
            y_start=legend_y,
            spacing=legend_spacing,
            font_size=font_size_captions,
        )

        plots_io_class.save_plotly_figure(
            fig,
            output_file,
            width=2480,
            height=max(C.BASE_HEIGHT_PER_ROW * rows, C.BASE_HEIGHT_PER_ROW),
            scale=C.SCALE,
            save_eps=False,
            save_final=True,
        )

    def plot_crossing_without_traffic_light(
        self,
        df_mapping,
        font_size_captions=40,
        x_axis_title_height=150,
        legend_x=0.92,
        legend_y=0.015,
        legend_spacing=0.02,
    ):
        """
        Plot locality crossing events without traffic equipment.

        Correct results.pickle index:
            30 = crossings_without_traffic_equipment_locality
        """
        data_tuple = self._load_results()
        without_trf_light = data_tuple[30]

        self._plot_traffic_equipment(
            df_mapping=df_mapping,
            source=without_trf_light,
            value_prefix="without_trf_light",
            axis_title="Road crossings without traffic signals (normalised)",
            output_file="crossings_without_traffic_equipment_avg",
            font_size_captions=font_size_captions,
            x_axis_title_height=x_axis_title_height,
            legend_x=legend_x,
            legend_y=legend_y,
            legend_spacing=legend_spacing,
        )

    def plot_crossing_with_traffic_light(
        self,
        df_mapping,
        font_size_captions=40,
        x_axis_title_height=150,
        legend_x=0.92,
        legend_y=0.015,
        legend_spacing=0.02,
    ):
        """
        Plot locality crossing events with traffic equipment.

        Correct results.pickle index:
            29 = crossings_with_traffic_equipment_locality
        """
        data_tuple = self._load_results()
        with_trf_light = data_tuple[29]

        self._plot_traffic_equipment(
            df_mapping=df_mapping,
            source=with_trf_light,
            value_prefix="with_trf_light",
            axis_title="Road crossings with traffic signals (normalised)",
            output_file="crossings_with_traffic_equipment_avg",
            font_size_captions=font_size_captions,
            x_axis_title_height=x_axis_title_height,
            legend_x=legend_x,
            legend_y=legend_y,
            legend_spacing=legend_spacing,
        )
