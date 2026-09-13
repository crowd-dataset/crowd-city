import common
import math
from tqdm import tqdm
import itertools
import pickle
import numpy as np
from custom_logger import CustomLogger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.plotting.layout import Layout
from utils.plotting import constants as C
from utils.core.iso import ISO
from utils.core.metadata import MetaData
from utils.core.grouping import Grouping
from utils.plotting.io import IO

layout_class = Layout()
iso_class = ISO()
metadata_class = MetaData()
grouping_class = Grouping()
plots_io_class = IO()
logger = CustomLogger(__name__)  # use custom logger

# File to store the locality coordinates
file_results = 'results.pickle'


class Stacked:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _mean_and_delta(day_value, night_value):
        """Return the mean of the available conditions and the night-day shift.

        A condition counts as present only when it carries a positive value.
        The day and night views substitute zeros for the condition they do not
        show, and a mean crossing speed of exactly zero is never a real
        observation, so both are treated as absent.
        """
        def usable(value):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            return numeric if math.isfinite(numeric) and numeric > 0.0 else None

        day = usable(day_value)
        night = usable(night_value)
        present = [value for value in (day, night) if value is not None]
        if not present:
            return None, None
        mean_value = sum(present) / len(present)
        delta = night - day if day is not None and night is not None else None
        return mean_value, delta

    def _add_mean_delta_row(self, fig, row, col, label_prefix, day_value,
                            night_value, marker_offset, label_suffix="",
                            bar_colour=None):
        """Draw one category as a mean bar plus a day/night shift marker.

        Day and night were previously stacked, which made the bar length
        day + night. That is not a meaningful quantity for two mean speeds: a
        city measured only by day looked identical to one whose day and night
        means were each half as large, while the printed label showed the
        mean. The bar now matches the number printed beside it, and a triangle
        past its end shows which condition is faster.
        """
        mean_value, delta = self._mean_and_delta(day_value, night_value)
        if mean_value is None:
            return None
        label = f"{label_prefix} {mean_value:.2f}"
        if label_suffix:
            label = f"{label_prefix} {label_suffix}"
        elif delta is not None:
            label = f"{label} ({delta:+.2f})"

        fig.add_trace(go.Bar(
            x=[mean_value],
            y=[label],
            orientation="h",
            name=f"{label_prefix} mean",
            marker=dict(color=bar_colour or C.BAR_COLOR_1),
            text=[""],
            textposition="inside",
            insidetextanchor="start",
            showlegend=False,
            textfont=dict(size=14, color="white"),
            hovertemplate=f"mean {mean_value:.2f}<extra></extra>",
        ), row=row, col=col)

        if delta is None or delta == 0.0:
            return label

        fig.add_trace(go.Scatter(
            x=[mean_value + marker_offset],
            y=[label],
            mode="markers",
            marker=dict(
                symbol="triangle-right" if delta > 0 else "triangle-left",
                size=18,
                color=C.BAR_COLOR_2,
                line=dict(width=0),
            ),
            showlegend=False,
            hovertemplate=f"night - day = {delta:+.2f}<extra></extra>",
        ), row=row, col=col)
        return label

    @staticmethod
    def _country_label_suffix(mean_value, delta, day_value, night_value,
                              all_sd_value, day_sd_value, night_sd_value):
        """Label for a country row, keeping the day and night detail visible."""
        if delta is not None:
            return (
                f"{mean_value:.2f}±{all_sd_value:.2f} "
                f"(D={float(day_value):.2f}±{day_sd_value:.2f}, "
                f"N={float(night_value):.2f}±{night_sd_value:.2f})"
            )
        try:
            day_present = float(day_value) > 0.0
        except (TypeError, ValueError):
            day_present = False
        sd_value = day_sd_value if day_present else night_sd_value
        condition = "D" if day_present else "N"
        return f"{mean_value:.2f}±{sd_value:.2f} ({condition} only)"

    @staticmethod
    def _tint(hex_colour, towards_white=0.55):
        """Blend a colour towards white.

        The city name is printed inside its own bar in black, so the saturated
        continent colours are lightened until that text stays legible while the
        hue still identifies the continent.
        """
        text = str(hex_colour).lstrip("#")
        if len(text) != 6:
            return hex_colour
        try:
            channels = [int(text[i:i + 2], 16) for i in (0, 2, 4)]
        except ValueError:
            return hex_colour
        blended = [
            round(value + (255 - value) * towards_white) for value in channels
        ]
        return "#%02x%02x%02x" % tuple(blended)

    @classmethod
    def _continent_colour(cls, continent):
        """Colour for a continent, shared with the other continent figures."""
        base = C.CONTINENT_COLORS.get(
            str(continent), C.CONTINENT_FALLBACK_COLOR
        ) if continent else C.CONTINENT_FALLBACK_COLOR
        return cls._tint(base)

    @staticmethod
    def _tick_step(max_value, target_ticks=8):
        """A round tick interval covering the axis from zero.

        Plotly's automatic ticks were starting partway along the axis, so the
        low values were never labelled even though the bars begin at zero.
        """
        if not max_value or max_value <= 0:
            return None
        raw = float(max_value) / float(target_ticks)
        magnitude = 10 ** math.floor(math.log10(raw))
        for multiple in (1, 2, 2.5, 5, 10):
            step = multiple * magnitude
            if step >= raw:
                return step
        return 10 * magnitude

    def _row_extent(self, day_values, night_values, count):
        """Largest mean across rows, with headroom for the shift marker."""
        means = []
        for index in range(count):
            mean_value, _ = self._mean_and_delta(
                day_values[index], night_values[index]
            )
            if mean_value is not None:
                means.append(mean_value)
        largest = max(means) if means else 0.0
        return largest * 1.10, largest * 0.025

    def stack_plot(self, df_mapping, order_by, metric, data_view, title_text, filename, analysis_level="locality",
                   font_size_captions=40, x_axis_title_height=110, legend_x=0.92, legend_y=0.015, legend_spacing=0.02,
                   left_margin=10, right_margin=10, columns=3):
        """
        Plots a stacked bar graph based on the provided data and configuration.

        Parameters:
            df_mapping (dict): A dictionary mapping categories to their respective DataFrames.
            order_by (str): Criterion to order the bars, e.g., 'alphabetical' or 'average'.
            metric (str): The metric to visualise, such as 'speed' or 'time'.
            data_view (str): Determines which subset of data to visualise, such as 'day', 'night', or 'combined'.
            title_text (str): The title of the plot.
            filename (str): The name of the file to save the plot as.
            font_size_captions (int, optional): Font size for captions. Default is 40.
            x_axis_title_height (int, optional): Vertical space for x-axis title. Default is 110.
            legend_x (float, optional): X position of the legend. Default is 0.92.
            legend_y (float, optional): Y position of the legend. Default is 0.015.
            legend_spacing (float, optional): Spacing between legend entries. Default is 0.02.

        Returns:
            None
        """

        # Define log messages in a structured way
        log_messages = {
            ("alphabetical", "speed", "day"): "Plotting speed to cross by alphabetical order during day time.",
            ("alphabetical", "speed", "night"): "Plotting speed to cross by alphabetical order during night time.",
            ("alphabetical", "speed", "combined"): "Plotting speed to cross by alphabetical order.",
            ("alphabetical", "time", "day"): "Plotting time to start cross by alphabetical order during day time.",
            ("alphabetical", "time", "night"): "Plotting time to start cross by alphabetical order during night time.",
            ("alphabetical", "time", "combined"): "Plotting time to start cross by alphabetical order.",
            ("average", "speed", "day"): "Plotting speed to cross by average during day time.",
            ("average", "speed", "night"): "Plotting speed to cross by averageduring night time.",
            ("average", "speed", "combined"): "Plotting speed to cross by average.",
            ("average", "time", "day"): "Plotting time to start cross by average during day time.",
            ("average", "time", "night"): "Plotting time to start cross by average during night time.",
            ("average", "time", "combined"): "Plotting time to start cross by average."
        }

        message = log_messages.get((order_by, metric, data_view))
        final_dict = {}

        if message:
            logger.info(message)

        # Map metric names to their index in the data tuple
        if analysis_level == "locality":
            metric_index_map = {
                "speed": 25,
                "time": 24
            }
        elif analysis_level == "country":
            metric_index_map = {
                "speed": 27,
                "time": 28
                }

        if metric not in metric_index_map:
            raise ValueError(f"Unsupported metric: {metric}")

        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        metric_data = data_tuple[metric_index_map[metric]]

        if metric_data is None:
            raise ValueError(f"'{metric}' returned None, please check the input data or calculations.")

        # Clean NaNs
        metric_data = {
            key: value for key, value in metric_data.items()
            if not (isinstance(value, float) and math.isnan(value))
        }

        def valid_metric_value(value) -> bool:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return False
            return math.isfinite(numeric) and numeric > 0.0

        def available_mean(locality: str) -> float:
            values = [
                float(final_dict[locality][f"{metric}_{condition}"])
                for condition in (0, 1)
                if valid_metric_value(
                    final_dict[locality].get(f"{metric}_{condition}")
                )
            ]
            return sum(values) / len(values) if values else 0.0

        if analysis_level == "locality":
            # Now populate the final_dict with locality-wise speed data
            for locality_condition, _ in tqdm(metric_data.items()):
                locality, lat, long, condition = locality_condition.split('_')

                # Get the country from the previously stored locality_country_map
                country = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "country")
                iso_code = metadata_class.get_value(df_mapping, "locality", locality, "lat", float(lat), "iso3")

                if country or iso_code is not None:
                    # Initialise the locality's dictionary if not already present
                    if f'{locality}_{lat}_{long}' not in final_dict:
                        continent = metadata_class.get_value(
                            df_mapping, "locality", locality, "lat",
                            float(lat), "continent",
                        )
                        final_dict[f'{locality}_{lat}_{long}'] = {f"{metric}_0": None,
                                                                  f"{metric}_1": None,
                                                                  "country": country,
                                                                  "iso": iso_code,
                                                                  "continent": continent}

                    # Populate the corresponding speed based on the condition
                    final_dict[f'{locality}_{lat}_{long}'][f"{metric}_{condition}"] = _

        if analysis_level == "country":
            for country_condition, _ in tqdm(metric_data.items()):
                country, condition = country_condition.split('_')

                # Get the iso3 from the mapping file
                iso_code = metadata_class.get_value(df=df_mapping,
                                                    column_name1="country",
                                                    column_value1=country,
                                                    column_name2=None,
                                                    column_value2=None,
                                                    target_column="iso3")

                if country is not None or iso_code is not None:
                    # Initialise the locality's dictionary if not already present
                    if f'{country}' not in final_dict:
                        final_dict[f'{country}'] = {f"{metric}_0": None, f"{metric}_1": None,
                                                    "country": country, "iso3": iso_code}
                    # Populate the corresponding speed based on the condition
                    final_dict[f'{country}'][f"{metric}_{condition}"] = _  # type: ignore

        if order_by == "alphabetical":
            if data_view == "day":
                cities_ordered = sorted(
                    [
                        locality for locality in final_dict.keys()
                        if valid_metric_value(final_dict[locality].get(f"{metric}_0"))
                    ],
                    key=lambda locality: (final_dict[locality].get("iso") or "")
                )
            elif data_view == "night":
                cities_ordered = sorted(
                    [
                        locality for locality in final_dict.keys()
                        if valid_metric_value(final_dict[locality].get(f"{metric}_1"))
                    ],
                    key=lambda locality: (final_dict[locality].get("iso") or "")
                )
            else:
                cities_ordered = sorted(
                    [
                        locality for locality in final_dict.keys()
                        if available_mean(locality) > 0.0
                    ],
                    key=lambda locality: (final_dict[locality].get("iso") or "")
                )

        elif order_by == "average":
            if data_view == "day":
                cities_ordered = sorted(
                    [
                        locality for locality in final_dict.keys()
                        if valid_metric_value(final_dict[locality].get(f"{metric}_0"))
                    ],
                    key=lambda locality: final_dict[locality].get(f"{metric}_0") or 0,
                    reverse=True
                )

            elif data_view == "night":
                cities_ordered = sorted(
                    [
                        locality for locality in final_dict.keys()
                        if valid_metric_value(final_dict[locality].get(f"{metric}_1"))
                    ],
                    key=lambda locality: final_dict[locality].get(f"{metric}_1") or 0,
                    reverse=True
                )

            else:
                cities_ordered = sorted(
                    [
                        locality for locality in final_dict.keys()
                        if available_mean(locality) > 0.0
                    ],
                    key=available_mean,
                    reverse=True,
                )

        if len(cities_ordered) == 0:
            logger.warning(
                f"Skipping {filename}: no finite positive {metric} values "
                f"are available for the {data_view} view."
            )
            return

        # Prepare data for day and night stacking
        day_key = f"{metric}_0"
        night_key = f"{metric}_1"

        if data_view == "combined":
            day_values = [final_dict[country][day_key] for country in cities_ordered]
            night_values = [final_dict[country][night_key] for country in cities_ordered]
        elif data_view == "day":
            day_values = [final_dict[country][day_key] for country in cities_ordered]
            night_values = [0 for country in cities_ordered]
        elif data_view == "night":
            day_values = [0 for country in cities_ordered]
            night_values = [final_dict[country][night_key] for country in cities_ordered]

        # Split the cities across the requested number of columns. The
        # trailing column simply holds fewer cities when the split is uneven.
        column_count = max(1, int(columns))
        num_cities_per_col = -(-len(cities_ordered) // column_count)
        column_slices = [
            cities_ordered[
                index * num_cities_per_col:(index + 1) * num_cities_per_col
            ]
            for index in range(column_count)
        ]

        # Define a base height per row and calculate total figure height
        # 58 px per row keeps the 22 px city label comfortable while the bars
        # still sit flush, so no white space separates neighbouring cities.
        # Adjust this single value to make the figure taller or shorter.
        TALL_FIG_HEIGHT = max(
            500,
            min(20000, 220 + num_cities_per_col * 58),
        )
        # Each column needs its own horizontal room, so the width follows the
        # column count rather than the old fixed two-column span.
        FIG_WIDTH = max(1200, min(4200, column_count * 1200))
        logger.info(
            f"Building {filename} for {len(cities_ordered)} cities "
            f"at {FIG_WIDTH}x{TALL_FIG_HEIGHT} pixels."
        )

        fig = make_subplots(
            rows=num_cities_per_col, cols=column_count,
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_cities_per_col),
        )

        # Bar length is the mean of the available conditions, so the axis and
        # the marker offset are derived from those means before drawing.
        max_value, marker_offset = self._row_extent(
            day_values, night_values, len(cities_ordered)
        )

        # Draw every column with one loop so the layout is not tied to a
        # fixed number of columns.
        for column_index, column_cities in enumerate(column_slices):
            column_number = column_index + 1
            offset = column_index * num_cities_per_col
            for i, locality in enumerate(column_cities):
                bar_colour = self._continent_colour(
                    final_dict[locality].get("continent")
                )
                locality_new, lat, long = locality.split('_')
                locality = grouping_class.process_locality_string(locality, df_mapping)

                if order_by == "average":
                    iso_code = metadata_class.get_value(
                        df_mapping, "locality", locality_new, "lat", float(lat), "iso3"
                    )
                    locality = iso_class.iso2_to_flag(
                        iso_class.iso3_to_iso2(iso_code)
                    ) + " " + locality  # type: ignore

                self._add_mean_delta_row(
                    fig=fig,
                    row=i + 1,
                    col=column_number,
                    label_prefix=locality,
                    day_value=day_values[offset + i],
                    night_value=night_values[offset + i],
                    marker_offset=marker_offset,
                    bar_colour=bar_colour,
                )

        # The x-axis spans the largest mean, not the largest day + night sum.

        # The bottom row differs per column when the split is uneven.
        last_rows = [max(1, len(chunk)) for chunk in column_slices]

        # Top row of every column carries the axis labels, bottom row repeats
        # them underneath.
        for column_index in range(column_count):
            column_number = column_index + 1
            last_row = last_rows[column_index]
            for i in range(1, num_cities_per_col + 1):
                if i % 2 == 1:
                    fig.update_xaxes(
                        range=[0, max_value],
                        row=i,
                        col=column_number,
                        showticklabels=(i == 1),
                        side='top', showgrid=False
                    )
                else:
                    fig.update_xaxes(
                        range=[0, max_value],
                        row=i,
                        col=column_number,
                        showticklabels=(i == last_row),
                        side='bottom', showgrid=False
                    )

        # Title and tick styling on the top row of each column, and the same
        # tick styling on each column's bottom row so both read alike.
        for column_index in range(column_count):
            column_number = column_index + 1
            fig.update_xaxes(
                title=dict(text=title_text,
                           font=dict(size=font_size_captions)),
                tickfont=dict(size=font_size_captions),
                ticks='outside',
                ticklen=10,
                tickwidth=2,
                tickcolor='black',
                row=1,
                col=column_number,
            )
            fig.update_xaxes(
                tickfont=dict(size=font_size_captions),
                ticks='outside',
                ticklen=10,
                tickwidth=2,
                tickcolor='black',
                row=last_rows[column_index],
                col=column_number,
            )

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Label the axis from zero. Without an explicit origin and step the
        # automatic ticks began partway along, leaving the low end unlabelled.
        tick_step = self._tick_step(max_value)
        if tick_step:
            fig.update_xaxes(tick0=0, dtick=tick_step)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='stack',
            height=TALL_FIG_HEIGHT,
            width=FIG_WIDTH,
            showlegend=False,  # Hide the default legend
            margin=dict(t=150, b=150),
            bargap=0,
            bargroupgap=0
        )

        # Define gridline generation parameters
        if metric == "speed":
            grid_step = 0.5
        elif metric == "time":
            grid_step = 1.0
        else:
            grid_step = 1.0

        # Gridlines every half unit for speed, matching the country figure, so
        # 0.5 and 1.5 get the same reference line as 1. Generated from the axis
        # maximum rather than a fixed count, so none are drawn off-range.
        x_grid_values = [
            grid_step * step_index
            for step_index in range(1, int(max_value / grid_step) + 1)
        ] if grid_step > 0 and max_value > 0 else []

        # Gridlines are drawn per column against that column's own x-axis,
        # which is 'x' for the first column and 'x2', 'x3', ... after it.
        for column_index in range(column_count):
            axis_ref = 'x' if column_index == 0 else 'x%d' % (column_index + 1)
            for x in x_grid_values:
                fig.add_shape(
                    type="line",
                    x0=x,
                    y0=0,
                    x1=x,
                    y1=1,
                    xref=axis_ref,
                    yref='paper',
                    line=dict(color="darkgray", width=1),
                    layer="above",
                )

        # Bars are coloured by continent, so the legend names the continents
        # actually plotted. The shift markers only exist in the combined view.
        plotted_continents = []
        for city_key in cities_ordered:
            continent_name = final_dict[city_key].get("continent")
            if continent_name and continent_name not in plotted_continents:
                plotted_continents.append(continent_name)

        legend_items = [
            {"name": str(name), "color": self._continent_colour(name)}
            for name in sorted(plotted_continents)
        ]
        if data_view == "combined":
            legend_items += [
                {"name": "▶ night faster", "color": C.BAR_COLOR_2},
                {"name": "◀ day faster", "color": C.BAR_COLOR_2},
            ]

        if legend_items:

            # Add the vertical legends at the top and bottom
            # Items are laid out downwards from y_start, so start high enough
            # that the last one still lands on legend_y instead of below the
            # figure. The two-item legend used to lose its second entry here.
            # legend_spacing is a fraction of figure height, so on a very
            # tall figure a fixed fraction pushes the entries far apart. Cap
            # it at a readable multiple of the text size.
            entry_spacing = min(
                legend_spacing,
                (font_size_captions * 2.0) / max(fig.layout.height or 1, 1),
            )
            layout_class.add_vertical_legend_annotations(fig,
                                                         legend_items,
                                                         x_position=min(legend_x, 0.80),
                                                         y_start=(
                                                             legend_y
                                                             + (len(legend_items) - 1)
                                                             * entry_spacing
                                                         ),
                                                         spacing=entry_spacing,
                                                         font_size=font_size_captions)

        # Box each column using the domain plotly actually assigned, so the
        # borders follow the column count instead of fixed halves.
        column_domains = []
        for column_index in range(column_count):
            axis_name = (
                'xaxis' if column_index == 0 else 'xaxis%d' % (column_index + 1)
            )
            domain = fig.layout[axis_name].domain
            column_domains.append((float(domain[0]), float(domain[1])))
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=domain[0],
                y0=1,
                x1=domain[1],
                y1=0.0,
                line=dict(color="black", width=2),
            )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_locality_map = {}
        for locality, info in final_dict.items():
            country = info['iso']  # type: ignore
            if country not in country_locality_map:
                country_locality_map[country] = []
            country_locality_map[country].append(locality)

        if order_by == "alphabetical":
            # One label pass per column, anchored just outside that
            # column's own left edge.
            font_size = C.FLAG_SIZE

            for column_index, column_cities in enumerate(column_slices):
                if not column_cities:
                    continue
                domain_start = column_domains[column_index][0]
                y_position_map = {}
                for row_index, locality in enumerate(column_cities):
                    country_code = final_dict[locality]['iso']
                    if country_code not in y_position_map:
                        y_position_map[country_code] = 1 - (
                            (row_index + 0.5) / max(len(column_cities), 1)
                        )

                for country_code, y_position in y_position_map.items():
                    iso2 = iso_class.iso3_to_iso2(country_code)
                    label = str(country_code) + iso_class.iso2_to_flag(iso2)
                    fig.add_annotation(
                        x=domain_start,
                        y=y_position,
                        xref="paper",
                        yref="paper",
                        text=label,
                        showarrow=False,
                        font=dict(size=font_size, color="black"),
                        xanchor='right',
                        align='right',
                        bgcolor='rgba(255,255,255,0.8)',
                    )

        fig.update_yaxes(
            tickfont=dict(size=C.TEXT_SIZE, color="black"),
            showticklabels=True,  # Ensure locality names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
            # Labels drawn inside the bars otherwise suppress any x-axis tick
            # they overlap, which silently dropped the low values: the left
            # column started at 0.5 and the right, with longer names, at 0.8.
            ticklabeloverflow='allow',
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(
            l=80,
            r=80,
            t=x_axis_title_height,
            b=max(10, font_size_captions + 30),
        ))
        plots_io_class.save_plotly_figure(fig=fig,
                                          filename=filename,
                                          width=FIG_WIDTH,
                                          height=TALL_FIG_HEIGHT,
                                          scale=C.SCALE,
                                          save_eps=True,
                                          save_final=True)

    def stack_plot_country(self, df_mapping, order_by, metric, data_view, title_text, filename,
                           legend_x=0.87, legend_y=0.04, font_size_captions=40, raw=False, legend_spacing=0.02,
                           left_margin=10, right_margin=10, top_margin=0, bottom_margin=0, height=2400, width=2480):
        """
        Plots a stacked bar graph based on the provided data and configuration.

        Parameters:
            df_mapping (dict): A dictionary mapping categories to their respective DataFrames.
            order_by (str): Criterion to order the bars, e.g., 'alphabetical' or 'average'.
            metric (str): The metric to visualise, such as 'speed' or 'time'.
            data_view (str): Determines which subset of data to visualise, such as 'day', 'night', or 'combined'.
            title_text (str): The title of the plot.
            filename (str): The name of the file to save the plot as.
            font_size_captions (int, optional): Font size for captions. Default is 40.
            x_axis_title_height (int, optional): Vertical space for x-axis title. Default is 110.
            legend_x (float, optional): X position of the legend. Default is 0.92.
            legend_y (float, optional): Y position of the legend. Default is 0.015.
            legend_spacing (float, optional): Spacing between legend entries. Default is 0.02.

        Returns:
            None
        """

        # Define log messages in a structured way
        log_messages = {
            ("alphabetical", "speed", "day"): "Plotting speed to cross by alphabetical order during day time.",
            ("alphabetical", "speed", "night"): "Plotting speed to cross by alphabetical order during night time.",
            ("alphabetical", "speed", "combined"): "Plotting speed to cross by alphabetical order.",
            ("alphabetical", "time", "day"): "Plotting time to start cross by alphabetical order during day time.",
            ("alphabetical", "time", "night"): "Plotting time to start cross by alphabetical order during night time.",
            ("alphabetical", "time", "combined"): "Plotting time to start cross by alphabetical order.",
            ("average", "speed", "day"): "Plotting speed to cross by average during day time.",
            ("average", "speed", "night"): "Plotting speed to cross by averageduring night time.",
            ("average", "speed", "combined"): "Plotting speed to cross by average.",
            ("average", "time", "day"): "Plotting time to start cross by average during day time.",
            ("average", "time", "night"): "Plotting time to start cross by average during night time.",
            ("average", "time", "combined"): "Plotting time to start cross by average.",
            ("condition", "time", "combined"): "Plotting time to start cross sorted by day values.",
            ("condition", "speed", "combined"): "Plotting speed to cross sorted by day values."
        }

        message = log_messages.get((order_by, metric, data_view))
        final_dict = {}

        if message:
            logger.info(message)

        # Map metric names to their index in the data tuple
        metric_index_map = {
            "speed": 27,
            "time": 28,
            "all_speed_country": 38,
            "all_time_country": 39
        }

        if metric not in metric_index_map:
            raise ValueError(f"Unsupported metric: {metric}")

        with open(file_results, 'rb') as file:
            data_tuple = pickle.load(file)

        no_of_crossing = data_tuple[35]

        metric_data = data_tuple[metric_index_map[metric]]
        all_values = data_tuple[metric_index_map[f"all_{metric}_country"]]

        if metric_data is None:
            raise ValueError(f"'{metric}' returned None, please check the input data or calculations.")

        # Clean NaNs
        metric_data = {
            key: value for key, value in metric_data.items()
            if not (isinstance(value, float) and math.isnan(value))
        }

        for country_condition, metric_values in tqdm(metric_data.items()):
            if not raw:
                if no_of_crossing[country_condition] < common.get_configs("min_crossing_detect"):
                    continue
            country, condition = country_condition.split('_')

            # Get the iso3 from the mapping file
            iso_code = metadata_class.get_value(df=df_mapping,
                                                column_name1="country",
                                                column_value1=country,
                                                column_name2=None,
                                                column_value2=None,
                                                target_column="iso3")

            if country is not None and iso_code is not None:
                # Initialise the country's dictionary if not already present
                if f'{country}' not in final_dict:
                    final_dict[f'{country}'] = {f"{metric}_0": None,
                                                f"{metric}_1": None,
                                                f"{metric}_sd_0": None,
                                                f"{metric}_sd_1": None,
                                                f"{metric}_sd_avg": None,
                                                "country": country,
                                                "iso3": iso_code}

                # Populate the corresponding speed based on the condition
                final_dict[f'{country}'][f"{metric}_{condition}"] = metric_values
                final_dict[f'{country}'][f"{metric}_sd_{condition}"] = np.std(all_values[f"{country}_{condition}"])

                vals = [v for k, v in all_values.items() if k.startswith(f"{country}_") and v is not None]
                flat_vals = list(itertools.chain.from_iterable(vals))
                final_dict[country][f"{metric}_sd_avg"] = np.std(flat_vals)

        if order_by == "alphabetical":
            if data_view == "day":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_0") or 0) >= 0.005
                    ],
                    key=lambda country: (final_dict[country].get("iso") or "")
                )
            elif data_view == "night":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_1") or 0) >= 0.005
                    ],
                    key=lambda country: (final_dict[country].get("iso") or "")
                )
            else:
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (((final_dict[country].get(f"{metric}_0") or 0) + (final_dict[country].get(f"{metric}_1") or 0)) / 2) >= 0.005  # noqa:E501
                    ],
                    key=lambda country: (final_dict[country].get("iso") or "")
                )

        elif order_by == "average":
            if data_view == "day":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_0") or 0) >= 0.005
                    ],
                    key=lambda country: final_dict[country].get(f"{metric}_0") or 0,
                    reverse=True
                )

            elif data_view == "night":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_1") or 0) >= 0.005
                    ],
                    key=lambda country: final_dict[country].get(f"{metric}_1") or 0,
                    reverse=True
                )

            else:
                # Rank by the same mean the bar draws: the average over the
                # conditions actually present. Dividing by two regardless put
                # a day-only country at half its bar length, so it landed far
                # below countries whose bars were visibly shorter.
                def country_mean(country: str) -> float:
                    mean_value, _ = self._mean_and_delta(
                        final_dict[country].get(f"{metric}_0"),
                        final_dict[country].get(f"{metric}_1"),
                    )
                    return mean_value or 0.0

                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if country_mean(country) >= 0.005
                    ],
                    key=country_mean,
                    reverse=True,
                )

        elif order_by == "condition":
            if data_view == "combined":
                countries_ordered = sorted(
                    [
                        country for country in final_dict.keys()
                        if (final_dict[country].get(f"{metric}_0") is not None or final_dict[country].get(f"{metric}_1") is not None)  # noqa:E501
                        and ((final_dict[country].get(f"{metric}_0") or final_dict[country].get(f"{metric}_1") or 0) >= 0.005)  # noqa:E501
                    ],
                    key=lambda country: (
                        final_dict[country].get(f"{metric}_0")
                        if final_dict[country].get(f"{metric}_0") is not None
                        else final_dict[country].get(f"{metric}_1") or 0
                        ), reverse=True
                    )

        if len(countries_ordered) == 0:
            return

        # Prepare data for day and night stacking
        day_key = f"{metric}_0"
        night_key = f"{metric}_1"
        day_sd_key = f"{metric}_sd_0"
        night_sd_key = f"{metric}_sd_1"
        all_sd_key = f"{metric}_sd_avg"

        if data_view == "combined":
            day_values = [final_dict[country][day_key] for country in countries_ordered]
            night_values = [final_dict[country][night_key] for country in countries_ordered]
            day_sd = [final_dict[country][day_sd_key] for country in countries_ordered]
            night_sd = [final_dict[country][night_sd_key] for country in countries_ordered]
            all_sd = [final_dict[country][all_sd_key] for country in countries_ordered]

        elif data_view == "day":
            day_values = [final_dict[country][day_key] for country in countries_ordered]
            day_sd = [final_dict[country][day_sd_key] for country in countries_ordered]
            night_values = [0 for country in countries_ordered]
            night_sd = [0 for country in countries_ordered]
            all_sd = [final_dict[country][all_sd_key] for country in countries_ordered]

        elif data_view == "night":
            day_values = [0 for country in countries_ordered]
            day_sd = [0 for country in countries_ordered]
            night_values = [final_dict[country][night_key] for country in countries_ordered]
            night_sd = [final_dict[country][night_sd_key] for country in countries_ordered]
            all_sd = [final_dict[country][all_sd_key] for country in countries_ordered]

        # Determine how many countries will be in each column
        num_countries_per_col = len(countries_ordered) // 2 + len(countries_ordered) % 2  # Split cities

        # Define a base height per row and calculate total figure height
        # TALL_FIG_HEIGHT = num_countries_per_col * BASE_HEIGHT_PER_ROW

        fig = make_subplots(
            rows=num_countries_per_col, cols=2,  # Two columns
            vertical_spacing=0.0005,  # Reduce the vertical spacing
            horizontal_spacing=0.01,  # Reduce horizontal spacing between columns
            row_heights=[1.0] * (num_countries_per_col),
        )

        # Bar length is the mean of the available conditions, so the axis and
        # the marker offset come from those means before drawing.
        max_value, marker_offset = self._row_extent(
            day_values, night_values, len(countries_ordered)
        )

        # Plot left column (first half of countries)
        for i, country in enumerate(countries_ordered[:num_countries_per_col]):

            # locality = wrapper_class.process_locality_string(locality, df_mapping)
            iso_code = metadata_class.get_value(df_mapping, "country", country, None, None, "iso3")

            # build up textual label for left column
            country = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + country

            mean_value, delta = self._mean_and_delta(
                day_values[i], night_values[i]
            )
            if mean_value is not None:
                self._add_mean_delta_row(
                    fig=fig,
                    row=i + 1,
                    col=1,
                    label_prefix=country,
                    day_value=day_values[i],
                    night_value=night_values[i],
                    marker_offset=marker_offset,
                    label_suffix=self._country_label_suffix(
                        mean_value, delta,
                        day_values[i], night_values[i],
                        all_sd[i], day_sd[i], night_sd[i],
                    ),
                )

        # Plot right column (second half of cities)
        for i, country in enumerate(countries_ordered[num_countries_per_col:]):
            # locality = wrapper_class.process_locality_string(locality, df_mapping)
            iso_code = metadata_class.get_value(df_mapping, "country", country, None, None, "iso3")

            # build up textual label for left column
            country = iso_class.iso2_to_flag(iso_class.iso3_to_iso2(iso_code)) + " " + country

            idx = num_countries_per_col + i
            mean_value, delta = self._mean_and_delta(
                day_values[idx], night_values[idx]
            )
            if mean_value is not None:
                self._add_mean_delta_row(
                    fig=fig,
                    row=i + 1,
                    col=2,
                    label_prefix=country,
                    day_value=day_values[idx],
                    night_value=night_values[idx],
                    marker_offset=marker_offset,
                    label_suffix=self._country_label_suffix(
                        mean_value, delta,
                        day_values[idx], night_values[idx],
                        all_sd[idx], day_sd[idx], night_sd[idx],
                    ),
                )

        # Identify the last row for each column where the last locality is plotted
        # The figure has num_countries_per_col rows, so doubling these indices
        # pushed them past the last row and the bottom axis never showed its
        # tick labels at all.
        last_row_left_column = num_countries_per_col
        last_row_right_column = max(
            1,
            len(countries_ordered) - num_countries_per_col,
        )
        first_row_left_column = 1  # The first row in the left column
        first_row_right_column = 1  # The first row in the right column

        # Update the loop for updating x-axes based on max values for speed and time
        for i in range(1, num_countries_per_col * 2 + 1):  # Loop through all rows in both columns
            # Update x-axis for the left column
            if i % 2 == 1:  # Odd rows
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=1,
                    showticklabels=(i == first_row_left_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=1,
                    showticklabels=(i == last_row_left_column),
                    side='bottom', showgrid=False
                )

            # Update x-axis for the right column
            if i % 2 == 1:  # Odd rows
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=2,  # Use speed max value for top axis
                    showticklabels=(i == first_row_right_column),
                    side='top', showgrid=False
                )
            else:  # Even rows (representing time)
                fig.update_xaxes(
                    range=[0, max_value],
                    row=i,
                    col=2,  # Use time max value for bottom axis
                    showticklabels=(i == last_row_right_column),
                    side='bottom', showgrid=False
                )

        # Update both y-axes (for left and right columns) to hide the tick labels
        fig.update_yaxes(showticklabels=False)

        # Ensure no gridlines are shown on x-axes and y-axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Label the axis from zero. Without an explicit origin and step the
        # automatic ticks began partway along, leaving the low end unlabelled.
        tick_step = self._tick_step(max_value)
        if tick_step:
            fig.update_xaxes(tick0=0, dtick=tick_step)

        # Update layout to hide the main legend and adjust margins
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='stack',
            height=height,
            width=width,
            showlegend=False,  # Hide the default legend
            margin=dict(t=0, b=0),
            bargap=0,
            bargroupgap=0
        )

        # Define gridline generation parameters
        if metric == "speed":
            start, step, count = 0.5, 0.5, 9
        elif metric == "time":
            start, step, count = 2, 2, 30

        # Generate gridline positions
        x_grid_values = [start + i * step for i in range(count)]

        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x,
                y0=0,
                x1=x,
                y1=1,  # Set the position of the gridlines
                xref='x',
                yref='paper',  # Ensure gridlines span the whole chart (yref='paper' spans full height)
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )

        # Manually add gridlines using `shapes` for the right column (x-axis 'x2')
        for x in x_grid_values:
            fig.add_shape(
                type="line",
                x0=x,
                y0=0,
                x1=x,
                y1=1,  # Set the position of the gridlines
                xref='x2',
                yref='paper',  # Apply to right column (x-axis 'x2')
                line=dict(color="darkgray", width=1),  # Customize the appearance of the gridlines
                layer="above"  # Draw the gridlines above the bars
            )
        # Set the x-axis labels (title_text) only for the last row and the first row
        fig.update_xaxes(
            title=dict(text=title_text,
                       font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=1
        )

        fig.update_xaxes(
            title=dict(text=title_text,
                       font=dict(size=font_size_captions)),
            tickfont=dict(size=font_size_captions),
            ticks='outside',
            ticklen=10,
            tickwidth=2,
            tickcolor='black',
            row=1,
            col=2
        )

        # The bottom axis of each column also carries tick labels, but only the
        # top row was ever styled, so those values rendered at the default size
        # while the ones above them were large.
        for bottom_row, bottom_col in (
            (last_row_left_column, 1),
            (last_row_right_column, 2),
        ):
            fig.update_xaxes(
                tickfont=dict(size=font_size_captions),
                ticks='outside',
                ticklen=10,
                tickwidth=2,
                tickcolor='black',
                row=bottom_row,
                col=bottom_col,
            )

        if data_view == "combined":
            # Define the legend items
            legend_items = [
                {"name": "Day-night mean", "color": C.BAR_COLOR_1},
                {"name": "▶ night faster", "color": C.BAR_COLOR_2},
                {"name": "◀ day faster", "color": C.BAR_COLOR_2},
            ]

            # Add the vertical legends at the top and bottom
            # Items are laid out downwards from y_start, so start high enough
            # that the last one still lands on legend_y instead of below the
            # figure. The two-item legend used to lose its second entry here.
            # legend_spacing is a fraction of figure height, so on a very
            # tall figure a fixed fraction pushes the entries far apart. Cap
            # it at a readable multiple of the text size.
            entry_spacing = min(
                legend_spacing,
                (font_size_captions * 2.0) / max(fig.layout.height or 1, 1),
            )
            layout_class.add_vertical_legend_annotations(fig,
                                                         legend_items,
                                                         x_position=min(legend_x, 0.80),
                                                         y_start=(
                                                             legend_y
                                                             + (len(legend_items) - 1)
                                                             * entry_spacing
                                                         ),
                                                         spacing=entry_spacing,
                                                         font_size=font_size_captions)

        # Add a box around the first column (left side)
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=1,
            x1=0.495,
            y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Add a box around the second column (right side)
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0.505,
            y0=1,
            x1=1,
            y1=0.0,
            line=dict(color="black", width=2)  # Black border for the box
        )

        # Create an ordered list of unique countries based on the cities in final_dict
        country_locality_map = {}
        for locality, info in final_dict.items():
            country = info['iso3']  # type: ignore
            if country not in country_locality_map:
                country_locality_map[country] = []
            country_locality_map[country].append(locality)

        fig.update_yaxes(
            tickfont=dict(size=C.TEXT_SIZE, color="black"),
            showticklabels=True,  # Ensure locality names are visible
            ticklabelposition='inside',  # Move the tick labels inside the bars
            # Labels drawn inside the bars otherwise suppress any x-axis tick
            # they overlap, which silently dropped the low values: the left
            # column started at 0.5 and the right, with longer names, at 0.8.
            ticklabeloverflow='allow',
        )
        fig.update_xaxes(
            tickangle=0,  # No rotation or small rotation for the x-axis
        )

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Final adjustments and display
        fig.update_layout(margin=dict(l=left_margin,
                                      r=right_margin,
                                      t=top_margin,
                                      b=max(bottom_margin, font_size_captions + 30)))
        # Speed (main text)
        # fig.add_annotation(
        #     text="0.5",
        #     xref="paper", yref="paper",
        #     x=0.06, y=1.032,  # adjust these values to position the label above the plot
        #     showarrow=False,
        #     font=dict(
        #         size=common.get_configs("font_size")+28,
        #         family=common.get_configs("font_family")
        #     )
        # )

        # fig.add_annotation(
        #     text="1",
        #     xref="paper", yref="paper",
        #     x=0.142, y=1.032,  # adjust these values to position the label above the plot
        #     showarrow=False,
        #     font=dict(
        #         size=common.get_configs("font_size")+28,
        #         family=common.get_configs("font_family")
        #     )
        # )

        # Time (main text)
        fig.add_annotation(
            text="2",
            xref="paper", yref="paper",
            x=0.595, y=1.032,  # adjust these values to position the label above the plot
            showarrow=False,
            font=dict(
                size=common.get_configs("font_size")+28,
                family=common.get_configs("font_family")
            )
        )

        # # Time (appendix)
        # fig.add_annotation(
        #     text="1",
        #     xref="paper", yref="paper",
        #     x=0.576, y=1.07,  # adjust these values to position the label above the plot
        #     showarrow=False,
        #     font=dict(
        #         size=common.get_configs("font_size")+28,
        #         family=common.get_configs("font_family")
        #     )
        # )

        plots_io_class.save_plotly_figure(fig=fig,
                                          filename=filename,
                                          width=width,
                                          height=height,
                                          scale=C.SCALE,
                                          save_eps=True,
                                          save_final=True)
