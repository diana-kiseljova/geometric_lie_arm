import io
import typing as ty

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image


# =========================
# Publication / Q1 style
# =========================
PAPER_RC = {
    # figure sizes
    "figure.figsize": (3.35, 2.6),   # близко к single-column
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.pad_inches": 0.03,

    # fonts
    "font.family": "serif",
    "mathtext.fontset": "stix",
    # если статья в LaTeX и хочешь полное совпадение шрифтов:
    # "text.usetex": True,

    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    # line widths
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,

    # clean look
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,

    # font embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
mpl.rcParams.update(PAPER_RC)


COLORS = {
    "actual": "black",
    "desired": "0.72",

    # muted x/y/z colors for local frames
    "x": "#C44E52",
    "y": "#55A868",
    "z": "#4C72B0",

    "grid3d": (0.80, 0.80, 0.80, 0.45),
    "grid2d": (0.70, 0.70, 0.70, 0.22),
}


ERROR_YLIMS = {

    # ошибки по времени
    "Orientation error, $e_{\theta}$ [rad]": (0.0, 2.0),
    "Position error, $e_{\rho}$ [m]": (0.0, 2.0),

    # компоненты psi
    "rotation_error_components": (-1.5, 1.5),
    "translation_error_components": (-1.5, 1.5)

}


def apply_fixed_ylim(ax, plot_key: str):
    """
    Applies fixed y-axis limits for error/control plots.

    plot_key must be one of ERROR_YLIMS keys.
    """
    ylim = ERROR_YLIMS.get(plot_key)

    if ylim is None:
        return

    ax.set_ylim(ylim[0], ylim[1])


class Painter:
    def __init__(
        self,
        T: float,
        figsize=(5.2, 3.2),
        local_coords_scale=0.12,
        title="",
        save_path: str | None = None,
        paired_frame_origin: str = "desired",   # "desired" | "actual" | "midpoint"
    ):
        self.T = T
        self.fig = plt.figure(figsize=figsize)
        self.axes = self.fig.add_subplot(111, projection="3d")

        self.local_coords_scale = local_coords_scale
        self.title = title
        self.save_path = save_path
        self.paired_frame_origin = paired_frame_origin

        self.fig.patch.set_facecolor("white")
        self.axes.set_facecolor("white")

    def _get_R_list(self, X_list: list[np.ndarray]) -> list[np.ndarray]:
        return [matrix[:3, :3] for matrix in X_list]

    def _get_p_nparray(self, X_list: list[np.ndarray]) -> np.ndarray:
        p_list = [matrix[:3, 3] for matrix in X_list]
        return np.vstack(p_list)

    def _plot_segment(
        self,
        ax,
        p0: np.ndarray,
        p1: np.ndarray,
        color: str,
        linestyle: str = "-",
        linewidth: float = 1.0,
        alpha: float = 1.0,
    ):
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    def _draw_frame(
        self,
        ax,
        R: np.ndarray,
        p: np.ndarray,
        linestyle: str = "-",
        linewidth: float = 1.0,
        alpha: float = 1.0,
        colors: tuple[str, str, str] | None = None,
    ):
        """
        Draw local coordinate frame with muted x/y/z colors.
        """
        if colors is None:
            colors = (COLORS["x"], COLORS["y"], COLORS["z"])

        for i in range(3):
            end = p + R[:, i] * self.local_coords_scale
            self._plot_segment(
                ax=ax,
                p0=p,
                p1=end,
                color=colors[i],
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

    def _int_only_formatter(self, x, pos):
        if np.isclose(x, round(x), atol=1e-8):
            return f"{int(round(x))}"
        return ""

    def _set_compact_axes(self, p_arrays: list[np.ndarray], margin_ratio: float = 0.06):
        all_points = np.vstack(p_arrays)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        ranges = maxs - mins

        # avoid zero range
        ranges[ranges < 1e-9] = 1.0
        margins = ranges * margin_ratio

        self.axes.set_xlim(mins[0] - margins[0], maxs[0] + margins[0])
        self.axes.set_ylim(mins[1] - margins[1], maxs[1] + margins[1])
        self.axes.set_zlim(mins[2] - margins[2], maxs[2] + margins[2])

        try:
            self.axes.set_box_aspect(ranges)
        except Exception:
            pass

    def _style_axes(self):
        ax = self.axes

        # axis labels
        ax.set_xlabel(r"$x$", labelpad=2)
        ax.set_ylabel(r"$y$", labelpad=2)
        ax.set_zlabel(r"$z$", labelpad=2)

        # z-label in 3D is often unreliable, so we place it robustly
        # ax.set_zlabel("")
        # ax.text2D(0.02, 0.92, r"$z$", transform=ax.transAxes)

        # fewer ticks, cleaner paper look
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

        formatter = FuncFormatter(self._int_only_formatter)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.zaxis.set_major_formatter(formatter)

        ax.tick_params(axis="both", which="major", labelsize=7, pad=1)
        try:
            ax.zaxis.set_tick_params(labelsize=7, pad=1)
        except Exception:
            pass

        # 3D grid styling
        ax.grid(True)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            try:
                axis._axinfo["grid"]["linewidth"] = 0.4
                axis._axinfo["grid"]["linestyle"] = "-"
                axis._axinfo["grid"]["color"] = COLORS["grid3d"]
            except Exception:
                pass

        # remove pane fills
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
            ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
            ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
        except Exception:
            pass

        # orthographic projection often looks cleaner in papers
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

    def _finalize_plot(self, p_arrays: list[np.ndarray]):
        self._set_compact_axes(p_arrays)
        self._style_axes()

        self.axes.view_init(elev=24, azim=-58)

        self.axes.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            bbox_transform=self.axes.transAxes,
            handlelength=2.2,
            borderaxespad=0.0,
        )

    def _choose_pair_origin(
        self,
        p_actual: np.ndarray,
        p_desired: np.ndarray,
    ) -> np.ndarray:
        if self.paired_frame_origin == "actual":
            return p_actual
        if self.paired_frame_origin == "midpoint":
            return 0.5 * (p_actual + p_desired)
        return p_desired  # default

    def _plot_trajectory_line(
        self,
        p_array: np.ndarray,
        label: str,
        line_color: str,
        line_style: str,
        line_width: float,
    ):
        self.axes.plot(
            p_array[:, 0],
            p_array[:, 1],
            p_array[:, 2],
            color=line_color,
            linestyle=line_style,
            linewidth=line_width,
            label=label,
        )

    def _plot_paired_frames(
        self,
        X_actual: list[np.ndarray],
        X_desired: list[np.ndarray],
        frame_every: int = 150,
        frame_width_actual: float = 0.95,
        frame_width_desired: float = 0.90,
    ):
        """
        Paired drawing by the same indices,
        but each frame is drawn at its own position:
        - actual  -> p_actual[i]
        - desired -> p_desired[i]
        """
        R_actual = self._get_R_list(X_actual)
        R_desired = self._get_R_list(X_desired)
        p_actual = self._get_p_nparray(X_actual)
        p_desired = self._get_p_nparray(X_desired)

        n = min(len(X_actual), len(X_desired))
        if n == 0:
            return

        indices = list(range(0, n, frame_every))
        if (n - 1) not in indices:
            indices.append(n - 1)

        for i in indices:
            # actual frame at actual position
            self._draw_frame(
                self.axes,
                R_actual[i],
                p_actual[i],
                linestyle="-",
                linewidth=frame_width_actual,
                alpha=0.95,
            )

            # desired frame at desired position
            self._draw_frame(
                self.axes,
                R_desired[i],
                p_desired[i],
                linestyle="--",
                linewidth=frame_width_desired,
                alpha=0.70,
            )

    def plot_2_trajectory(
        self,
        X_actual: list[np.ndarray],
        X_desired: list[np.ndarray],
        label_actual: str = "actual",
        label_desired: str = "desired",
    ):
        self.axes.cla()

        n = min(len(X_actual), len(X_desired))
        X_actual = X_actual[:n]
        X_desired = X_desired[:n]

        p_actual = self._get_p_nparray(X_actual)
        p_desired = self._get_p_nparray(X_desired)

        # actual trajectory
        self._plot_trajectory_line(
            p_actual,
            label=label_actual,
            line_color=COLORS["actual"],
            line_style="-",
            line_width=1.35,
        )

        # desired trajectory
        self._plot_trajectory_line(
            p_desired,
            label=label_desired,
            line_color=COLORS["desired"],
            line_style="--",
            line_width=1.20,
        )

        # paired local frames
        self._plot_paired_frames(
            X_actual=X_actual,
            X_desired=X_desired,
            frame_every=150,
            frame_width_actual=0.95,
            frame_width_desired=0.90,
        )

        self._finalize_plot([p_actual, p_desired])

        if self.save_path is not None:
            base = f"{self.save_path}/trajectory"
            self.fig.savefig(
                base + ".pdf",
                dpi=600,
                facecolor="white",
                edgecolor="none",
                bbox_inches="tight",     # bbox_inches=None #for 3D this is often safer
                pad_inches=0.02
            )
            self.fig.savefig(
                base + ".png",
                dpi=600,
                facecolor="white",
                edgecolor="none",
                bbox_inches="tight",
                pad_inches=0.02
            )

    def _fig_to_img(self, fig, dpi=300, transparent=False) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=transparent,
        )
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img

    def _apply_2d_axes_style(self, ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=COLORS["grid2d"][3])
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3, width=0.7)

    def _autoscale_y_with_margin(self, ax, y: np.ndarray, margin_ratio: float = 0.08):
        y = np.asarray(y)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return

        y_min = float(np.min(y))
        y_max = float(np.max(y))

        if np.isclose(y_min, y_max):
            delta = max(abs(y_min) * 0.05, 1e-3)
            ax.set_ylim(y_min - delta, y_max + delta)
            return

        margin = (y_max - y_min) * margin_ratio
        ax.set_ylim(y_min - margin, y_max + margin)

    def draw_one_error(
        self,
        error: np.ndarray,
        error_info: dict[str, float],
        label: str,
        img_name: str
    ) -> ty.Optional[Image.Image]:
        num_steps = len(error)
        time = np.linspace(0, self.T, num_steps)

        fig_err, ax = plt.subplots(figsize=(3.35, 2.2), constrained_layout=True)
        fig_err.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(time, error[:num_steps], color="black", linewidth=1.15, label=label)
        apply_fixed_ylim(ax, label)
        ax.set_xlabel(r"$t\;[\mathrm{s}]$")
        ax.set_ylabel(label)

        self._apply_2d_axes_style(ax)
        self._autoscale_y_with_margin(ax, error[:num_steps])

        if label:
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0
            )

        image_err = None
        if self.save_path is not None:
            base = f"{self.save_path}/error_{img_name}"
            fig_err.savefig(base + ".pdf", dpi=600, bbox_inches="tight", facecolor="white")
            fig_err.savefig(base + ".png", dpi=600, bbox_inches="tight", facecolor="white")
        else:
            image_err = self._fig_to_img(fig_err)

        plt.close(fig_err)
        return image_err

    def draw(
        self,
        X_desired: list[np.ndarray],
        X_actual: list[np.ndarray],
        psi_hist: list[np.ndarray],
    ) -> tuple[
        ty.Optional[Image.Image],
        ty.Optional[Image.Image],
        ty.Optional[Image.Image],
    ]:
        n = min(len(X_desired), len(X_actual))
        X_desired = X_desired[:n]
        X_actual = X_actual[:n]

        self.plot_2_trajectory(
            X_actual=X_actual,
            X_desired=X_desired,
            label_actual="real trajectory",
            label_desired="referecne trajectory",
        )

        time = np.linspace(0, self.T, n)
        psi_hist = np.vstack(psi_hist)[:n]

        # orientation errors
        fig_err_or, ax_or = plt.subplots(figsize=(3.35, 2.2), constrained_layout=True)
        fig_err_or.patch.set_facecolor("white")
        ax_or.set_facecolor("white")

        ax_or.plot(time, psi_hist[:, 0], label=r"$\theta_x$ [rad]", color=COLORS["x"], linewidth=1.05)
        ax_or.plot(time, psi_hist[:, 1], label=r"$\theta_y$ [rad]", color=COLORS["y"], linewidth=1.05)
        ax_or.plot(time, psi_hist[:, 2], label=r"$\theta_z$ [rad]", color=COLORS["z"], linewidth=1.05)
        apply_fixed_ylim(ax_or, "rotation_error_components")
        ax_or.set_xlabel(r"$t\;[\mathrm{s}]$")
        ax_or.set_ylabel("Orientation error [rad]")
        self._apply_2d_axes_style(ax_or)
        self._autoscale_y_with_margin(ax_or, psi_hist[:, :3].reshape(-1))
        ax_or.legend(
            loc="upper right",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0
        )

        # position errors
        fig_err_ps, ax_ps = plt.subplots(figsize=(3.35, 2.2), constrained_layout=True)
        fig_err_ps.patch.set_facecolor("white")
        ax_ps.set_facecolor("white")

        ax_ps.plot(time, psi_hist[:, 3], label=r"$\rho_x$ [m]", color=COLORS["x"], linewidth=1.05)
        ax_ps.plot(time, psi_hist[:, 4], label=r"$\rho_y$ [m]", color=COLORS["y"], linewidth=1.05)
        ax_ps.plot(time, psi_hist[:, 5], label=r"$\rho_z$ [m]", color=COLORS["z"], linewidth=1.05)
        apply_fixed_ylim(ax_ps, "translation_error_components")
        ax_ps.set_xlabel(r"$t\;[\mathrm{s}]$")
        ax_ps.set_ylabel("Position error [m]")
        self._apply_2d_axes_style(ax_ps)
        self._autoscale_y_with_margin(ax_ps, psi_hist[:, 3:6].reshape(-1))
        ax_ps.legend(
            loc="upper right",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0
        )

        image_traj, image_err_or, image_err_ps = None, None, None
        if self.save_path is not None:
            fig_err_or.savefig(
                f"{self.save_path}/error_or.pdf",
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )
            fig_err_or.savefig(
                f"{self.save_path}/error_or.png",
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )

            fig_err_ps.savefig(
                f"{self.save_path}/error_ps.pdf",
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )
            fig_err_ps.savefig(
                f"{self.save_path}/error_ps.png",
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )
        else:
            image_traj = self._fig_to_img(self.fig, dpi=300, transparent=False)
            image_err_or = self._fig_to_img(fig_err_or, dpi=300, transparent=False)
            image_err_ps = self._fig_to_img(fig_err_ps, dpi=300, transparent=False)

        plt.close(self.fig)
        plt.close(fig_err_or)
        plt.close(fig_err_ps)

        return image_traj, image_err_or, image_err_ps