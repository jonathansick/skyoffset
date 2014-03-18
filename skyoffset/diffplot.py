#!/usr/bin/env python
# encoding: utf-8
"""
Plot distributions of difference pixels.
"""

import os

import numpy as np
import astropy.io.fits
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec


def plot_diffs(mosaic_doc, plot_dir):
    """Make diff pixels histogram plots for all differences in the given
    mosaic document.
    
    Parameters
    ----------
    mosaic_doc : dict
        The document from MosaicDB for this mosaic.
    plot_dir : str
        Directory to save plots to.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for pair_key, diff in mosaic_doc['couplings']['diff_paths'].iteritems():
        median = mosaic_doc['couplings']['diffs'][pair_key]
        sigma = mosaic_doc['couplings']['sigmas'][pair_key]
        plot_path = os.path.join(plot_dir, pair_key)
        plot_diff(diff, median, sigma, plot_path)


def plot_diff(diff_path, median, sigma, plot_path):
    """Plot histogram of the difference image."""
    fits = astropy.io.fits.open(diff_path)
    pixels = fits[0].data
    pixels = pixels[np.isfinite(pixels)].ravel()

    fig = Figure(figsize=(3.5, 3.5))
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1, left=0.15, right=0.95, bottom=0.15, top=0.95,
        wspace=None, hspace=None, width_ratios=None, height_ratios=None)
    ax = fig.add_subplot(gs[0])
    ax.hist(pixels, 1000, histtype='stepfilled',
            edgecolor='None', facecolor='dodgerblue')
    ax.axvline(median, ls='-', c='k', lw=2)
    ax.axvline(median - sigma, ls='--', c='k', lw=1)
    ax.axvline(median + sigma, ls='--', c='k', lw=1)
    ax.text(0.1, 0.9, r"$%.2f \pm %.2f$" % (median, sigma),
            ha='left', va='top',
            transform=ax.transAxes)
    ax.set_xlim(median - 3 * sigma, median + 3 * sigma)
    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure(plot_path + ".pdf", format="pdf")

    fits.close()
