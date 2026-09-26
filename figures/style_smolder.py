"""Shared plotting style for every SMOLDER figure with real x/y axes.

House style, applied identically everywhere so the repo reads as one
document rather than a stitched-together set of one-off scripts:
  - light grey plot background (the panel between the axes)
  - white dashed gridlines at every major x AND y tick
  - no top/right spines, thin grey bottom/left spines
  - one shared colour palette (ACCENT = primary series, ACCENT2 = secondary)

Figures with no cartesian axes (the architecture diagram, the input-channel
panel) do not use this module; the rule is specifically about chart panels.
"""
import matplotlib.pyplot as plt

PANEL_BG = "#E9E9EC"
FIG_BG = "#FFFFFF"
GRID_COLOR = "#FFFFFF"
SPINE_COLOR = "#6B6B6E"
INK = "#1A202C"
ACCENT = "#1A6FAA"      # primary series (the incumbent model)
ACCENT2 = "#E07A1A"     # secondary series / highlight
MUTED = "#8C8C90"       # de-emphasised reference lines

FONT = "DejaVu Sans"


def style_axes(ax, grid_x=True, grid_y=True):
    """Apply the house style to one Axes: grey panel, white dashed gridlines
    at major ticks, clean spines. Call AFTER setting ticks/limits/scale."""
    ax.set_facecolor(PANEL_BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.9)
    ax.minorticks_off()          # every visible tick carries a gridline
    ax.tick_params(colors=INK, labelsize=9.5, length=3.5)
    if grid_x:
        ax.grid(axis="x", which="major", color=GRID_COLOR, linestyle="--",
                linewidth=1.0, alpha=0.95, zorder=0)
    if grid_y:
        ax.grid(axis="y", which="major", color=GRID_COLOR, linestyle="--",
                linewidth=1.0, alpha=0.95, zorder=0)
    ax.set_axisbelow(True)
    return ax


def new_figure(figsize):
    fig = plt.figure(figsize=figsize, facecolor=FIG_BG)
    return fig
