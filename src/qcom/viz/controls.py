"""
Plotting helpers for control time series.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Mapping, cast

import numpy as np


def plot_time_series(
    time_series,
    *,
    style: str = "auto",
    order: Iterable[str] | None = None,
    labels: Mapping[str, str] | None = None,
    y_hints: Mapping[str, tuple[float, float]] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """
    Visualize a TimeSeries-like object.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"

    presets = {
        "rydberg": {
            "order": ("omega", "delta", "phi"),
            "labels_abs": {
                "omega": r"$\Omega$",
                "delta": r"$\Delta$",
                "phi": r"$\Phi\,(\mathrm{rad})$",
            },
            "labels_norm": {
                "omega": r"$\Omega_{\mathrm{env}}$",
                "delta": r"$\Delta_{\mathrm{env}}$",
                "phi": r"$\Phi_{\mathrm{env}}$",
            },
            "norm_y_hints": {
                "omega": (-0.05, 1.05),
                "delta": (-1.05, 1.05),
            },
            "title_abs": "TimeSeries (Rydberg, Absolute Values)",
            "title_norm": "TimeSeries (Rydberg, Normalized Values)",
        },
    }

    channels = time_series.channels
    lower_to_orig = {k.lower(): k for k in channels.keys()}
    present_canon = list(lower_to_orig.keys())

    if not present_canon:
        fig, ax = plt.subplots(figsize=(figsize if figsize else (6.0, 2.0)))
        ax.axis("off")
        ax.set_title("TimeSeries (empty)")
        return fig, [ax]

    chosen_preset = None
    if style == "auto":
        if set(present_canon) == {"omega", "delta", "phi"}:
            chosen_preset = "rydberg"
    elif style == "per_channel":
        chosen_preset = None
    elif style.lower() in presets:
        chosen_preset = style.lower()
    else:
        raise ValueError(f"Unknown plot style/preset: {style!r}")

    if order is not None:
        desired = [str(n).lower() for n in order]
        canon_list = [c for c in desired if c in lower_to_orig]
    elif chosen_preset is not None:
        preset_order = presets[chosen_preset]["order"]
        canon_list = [c for c in preset_order if c in lower_to_orig]
    else:
        canon_list = list(present_canon)

    if not canon_list:
        fig, ax = plt.subplots(figsize=(figsize if figsize else (6.0, 2.0)))
        ax.axis("off")
        ax.set_title("TimeSeries (no matching channels)")
        return fig, [ax]

    labels = dict(labels or {})
    y_hints = dict(y_hints or {})

    if chosen_preset is not None:
        preset = presets[chosen_preset]
        default_labels = cast(
            Mapping[str, str],
            preset["labels_norm"] if time_series.mode == "normalized" else preset["labels_abs"],
        )
        for c in canon_list:
            if c not in labels and c in default_labels:
                labels[c] = default_labels[c]
        if time_series.mode == "normalized":
            norm_y_hints = cast(Mapping[str, tuple[float, float]], preset.get("norm_y_hints", {}))
            for c, rng in norm_y_hints.items():
                if c in canon_list and c not in y_hints:
                    y_hints[c] = rng

    for c in canon_list:
        labels.setdefault(c, c)

    tmin = min(channels[lower_to_orig[c]][0][0] for c in present_canon)
    tmax = max(channels[lower_to_orig[c]][0][-1] for c in present_canon)
    have_union = np.isfinite([tmin, tmax]).all() and (tmax >= tmin)

    n_axes = len(canon_list)
    if figsize is None:
        h = min(max(2.2 * n_axes, 2.0), 12.0)
        figsize = (8.0, h)

    fig, axes = plt.subplots(n_axes, 1, sharex=True, figsize=figsize, constrained_layout=False)
    if n_axes == 1:
        axes = [axes]

    for ax, c in zip(axes, canon_list):
        name_orig = lower_to_orig[c]
        t, v = channels[name_orig]

        ax.plot(t, v, "o", label="input values")
        ax.plot(t, v, "-", linewidth=1.5, label="input curve")

        if have_union and t.size:
            if t[0] > tmin:
                ax.hlines(0.0, tmin, t[0], linestyles="dashed", label="default outside domain")
            if t[-1] < tmax:
                ax.hlines(0.0, t[-1], tmax, linestyles="dashed", label="default outside domain")

        ax.grid(True, alpha=0.3)
        ax.set_ylabel(labels.get(c, c), fontsize=12)
        if c in y_hints:
            ymin, ymax = y_hints[c]
            ax.set_ylim(float(ymin), float(ymax))

    axes[-1].set_xlabel(r"time $t$ (s)", fontsize=12)
    if have_union:
        span = max(tmax - tmin, 1e-15)
        axes[-1].set_xlim(tmin - 0.02 * span, tmax + 0.02 * span)

    if chosen_preset is not None:
        preset = presets[chosen_preset]
        title_str = (
            preset["title_norm"] if time_series.mode == "normalized" else preset["title_abs"]
        )
    else:
        title_str = "TimeSeries (Per-Channel)" if style == "per_channel" else "TimeSeries"

    title_obj = fig.suptitle(title_str, fontsize=14, y=1.0, va="bottom")
    handles_labels = [ax.get_legend_handles_labels() for ax in axes]
    handles, texts = max(handles_labels, key=lambda hl: len(hl[1]))
    legend_obj = None
    if texts:
        seen = set()
        deduped_handles = []
        deduped_texts = []
        for handle, text in zip(handles, texts):
            if text not in seen:
                seen.add(text)
                deduped_handles.append(handle)
                deduped_texts.append(text)
        legend_obj = fig.legend(
            deduped_handles,
            deduped_texts,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            bbox_transform=fig.transFigure,
            ncol=min(3, len(deduped_texts)),
            frameon=True,
            fontsize=11,
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _h_in_figcoords(artist):
        bbox = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        return bbox.height

    title_h = _h_in_figcoords(title_obj) if title_obj is not None else 0.0
    legend_h = _h_in_figcoords(legend_obj) if legend_obj is not None else 0.0
    gap_title_legend = 0.010
    gap_legend_axes = 0.020
    top = 1.0 - (
        title_h
        + (legend_h if legend_obj else 0.0)
        + (gap_title_legend if legend_obj else 0.0)
        + gap_legend_axes
    )
    top = max(0.55, min(0.92, top))
    fig.subplots_adjust(top=top)

    return fig, axes
