import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
from scipy.stats import mstats
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mc
import warnings 
warnings.filterwarnings("ignore")


def lighten_color(color, amount=0.5, desaturation=0.2):
    """
    Eric's function.
    Lightens and desaturates the given color by multiplying (1-luminosity) by the given amount
    and decreasing the saturation by the specified desaturation amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    >> lighten_color('g', 0.3, 0.2)
    >> lighten_color('#F034A3', 0.6, 0.4)
    >> lighten_color((.3,.55,.1), 0.5, 0.1)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    new_luminosity = 1 - amount * (1 - c[1])
    new_saturation = max(0, c[2] - desaturation)
    return colorsys.hls_to_rgb(c[0], new_luminosity, new_saturation)

def get_fancy_bbox(bb, boxstyle, color, background=False, mutation_aspect=3, i=0):
    """
    Creates a fancy bounding box for the bar plots. Adapted from Eric's function.
    """
    if background:
        height = bb.height - 2
    else:
        height = bb.height
    if background:
        base = bb.ymin # - 0.2
    else:
        base = bb.ymin
    return FancyBboxPatch(
        (bb.xmin, base),
        abs(bb.width), height,
        boxstyle=boxstyle,
        ec="none", fc=color,
        mutation_aspect=mutation_aspect, # change depending on ylim
        zorder=2,
        # hatch = "" if (i + 1) % 2 == 1 else "/",

    )

def change_saturation(rgb, change=0.6):
    """
    Changes the saturation for the plotted bars, rgb is from sns.colorblind (used change=0.6 in paper)
    """
    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    saturation = max(0, min(hsv[1] * change, 1))
    return colorsys.hsv_to_rgb(hsv[0], saturation, hsv[2])
