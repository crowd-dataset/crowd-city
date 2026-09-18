import os
import shutil
import subprocess

import common
import plotly as py
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger


class IO:
    def __init__(self) -> None:
        pass

    def save_plotly_figure(self, fig, filename, width=1600, height=900, scale=1, save_final=True, save_png=True,
                           save_eps=True):
        """
        Saves a Plotly figure as HTML, PNG and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): width of the PNG and EPS images in pixels. Defaults to 1600.
            height (int, optional): height of the PNG and EPS images in pixels. Defaults to 900.
            scale (int, optional): Scaling factor for the PNG image. Defaults to 3.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # Raster export goes through kaleido, which launches a headless
        # Chromium. That step is prone to hanging indefinitely on Windows,
        # where it blocks the whole analysis on a figure rather than failing.
        # Setting save_images to false keeps the interactive HTML, which
        # carries the same data, and skips the PNG and EPS export.
        if not common.get_configs("save_images"):
            save_png = False
            save_eps = False

        # Create directory if it doesn't exist
        output_final = os.path.join(common.root_dir, 'figures')
        os.makedirs(common.output_dir, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)

        # Save as HTML
        logger.info(f"Saving html file for {filename}.")
        py.offline.plot(fig, filename=os.path.join(common.output_dir, filename + ".html"))
        # also save the final figure
        if save_final:
            py.offline.plot(fig, filename=os.path.join(output_final, filename + ".html"),  auto_open=False)

        try:
            # Save as PNG
            if save_png:
                logger.info(f"Saving png file for {filename}.")
                fig.write_image(os.path.join(common.output_dir, filename + ".png"), width=width, height=height,
                                scale=scale)
                # also save the final figure
                if save_final:
                    shutil.copy(os.path.join(common.output_dir, filename + ".png"),
                                os.path.join(output_final, filename + ".png"))

            # Save as EPS
            if save_eps:
                logger.info(f"Saving eps file for {filename}.")
                self._write_eps(fig, os.path.join(common.output_dir, filename + ".eps"), width, height)
                # also save the final figure
                if save_final:
                    shutil.copy(os.path.join(common.output_dir, filename + ".eps"),
                                os.path.join(output_final, filename + ".eps"))
        except ValueError as e:
            logger.error(f"Value error raised when attempted to save image {filename}: {e}")

    def _write_eps(self, fig, eps_path, width, height):
        """Write an EPS file, working around kaleido's PDF-to-EPS conversion.

        Kaleido bundles its own PDF-to-EPS step, which raises "Transform
        failed with error code 256: PDF to EPS conversion failed" on some
        systems regardless of figure content, even though kaleido's own PDF
        and PNG export both work fine there. Exporting to PDF and converting
        with poppler's pdftops sidesteps that broken step; if pdftops is not
        installed, this raises the same ValueError the caller already handles.
        """
        try:
            fig.write_image(eps_path, width=width, height=height)
            return
        except ValueError as error:
            logger.warning(
                f"Kaleido's direct EPS export failed ({error}); "
                "falling back to PDF + pdftops."
            )

        pdf_path = eps_path[:-len(".eps")] + ".pdf"
        fig.write_image(pdf_path, width=width, height=height)
        try:
            subprocess.run(
                ["pdftops", "-eps", pdf_path, eps_path],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise ValueError(f"pdftops fallback failed for {eps_path}: {error}") from error
        finally:
            os.remove(pdf_path)
