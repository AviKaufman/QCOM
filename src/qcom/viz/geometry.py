"""
Plotting helpers for lattice geometry.
"""

from __future__ import annotations

import numpy as np


def plot_lattice_register(register, show_index: bool = True, default_s: float = 200, **kwargs):
    """
    Visualize a LatticeRegister-like object.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from qcom._internal.fonts import publication_font_context

    with plt.rc_context(publication_font_context()):
        X = register.positions
        s = kwargs.pop("s", default_s)

        if X.shape[0] == 0:
            _, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "Empty LatticeRegister",
                ha="center",
                va="center",
                fontsize=16,
                color="gray",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            return ax

        if np.allclose(X[:, 2], 0.0):
            _, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], s=s, **kwargs)

            if show_index:
                font_sz = (s**0.5) * 0.6
                for i, (x, y, _) in enumerate(X):
                    ax.text(
                        x,
                        y,
                        str(i),
                        ha="center",
                        va="center",
                        fontsize=font_sz,
                        color="white",
                        weight="bold",
                    )

            x = X[:, 0]
            y = X[:, 1]
            xmin, xmax = float(x.min()), float(x.max())
            ymin, ymax = float(y.min()), float(y.max())
            xspan = xmax - xmin
            yspan = ymax - ymin

            tol_x = 1e-12 * max(1.0, abs(xmin), abs(xmax))
            tol_y = 1e-12 * max(1.0, abs(ymin), abs(ymax))
            near_zero_x = xspan <= tol_x
            near_zero_y = yspan <= tol_y

            frac_used = 0.05
            frac_unused = 0.015

            if near_zero_x and near_zero_y:
                x0 = 0.5 * (xmin + xmax)
                y0 = 0.5 * (ymin + ymax)
                pad = 1.0
                ax.set_xlim(x0 - pad, x0 + pad)
                ax.set_ylim(y0 - pad, y0 + pad)
                ax.set_aspect("auto")
            elif near_zero_x and not near_zero_y:
                x0 = 0.5 * (xmin + xmax)
                pad_x = max(
                    frac_unused * (yspan if yspan > 0 else 1.0),
                    tol_x if tol_x > 0 else 1.0,
                )
                ax.set_xlim(x0 - pad_x, x0 + pad_x)
                pad_y = frac_used * yspan if yspan > 0 else 1.0
                ax.set_ylim(ymin - pad_y, ymax + pad_y)
                ax.set_aspect("auto")
            elif near_zero_y and not near_zero_x:
                y0 = 0.5 * (ymin + ymax)
                pad_y = max(
                    frac_unused * (xspan if xspan > 0 else 1.0),
                    tol_y if tol_y > 0 else 1.0,
                )
                ax.set_ylim(y0 - pad_y, y0 + pad_y)
                pad_x = frac_used * xspan if xspan > 0 else 1.0
                ax.set_xlim(xmin - pad_x, xmax + pad_x)
                ax.set_aspect("auto")
            else:
                pad_x = frac_used * xspan if xspan > 0 else 1.0
                pad_y = frac_used * yspan if yspan > 0 else 1.0
                ax.set_xlim(xmin - pad_x, xmax + pad_x)
                ax.set_ylim(ymin - pad_y, ymax + pad_y)
                ax.set_aspect("equal")

            ax.set_title("Lattice Register (2D)", fontsize=16)
            ax.set_xlabel(r"$x$ (m)", fontsize=14)
            ax.set_ylabel(r"$y$ (m)", fontsize=14)
            return ax

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=s, **kwargs)

        if show_index:
            font_sz = (s**0.5) * 0.4
            for i, (x, y, z) in enumerate(X):
                ax.text(
                    x,
                    y,
                    z,
                    str(i),
                    ha="center",
                    va="center",
                    fontsize=font_sz,
                    color="black",
                )

        ax.set_title("Lattice Register (3D)", fontsize=16)
        ax.set_xlabel(r"$x$ (m)", fontsize=14)
        ax.set_ylabel(r"$y$ (m)", fontsize=14)
        ax.set_zlabel(r"$z$ (m)", fontsize=14)
        return ax
