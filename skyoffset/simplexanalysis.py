#!/usr/bin/env python
# encoding: utf-8
"""
Analyze the convergence performance of a multi-start reconverging
simplex run.

History
-------
2011-07-18 - Created by Jonathan Sick

"""

__all__ = ['']

import os
import pymongo

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def main():
    runName = "mosaic_solver_cython_J"
    analyzer = SimplexAnalyzer(runName, "skyoffsets/simplex_analyzer")
    analyzer.energy_hist()
    analyzer.extime_nrestarts()

class SimplexAnalyzer(object):
    """Analyze the convergence performance of multi-start re-converging
    simplex sky offset optimization."""
    def __init__(self, cname, workDir, dbname="skyoffsets", url="localhost",
            port=27017):
        super(SimplexAnalyzer, self).__init__()
        self.cname = cname
        self.dbname = dbname
        self.url = url
        self.port = port
        self.db = pymongo.Connection(self.url, self.port)[dbname]
        self.c = self.db[cname]
        
        self.workDir = workDir
        if os.path.exists(workDir) is False: os.makedirs(workDir)
    
    def energy_hist(self, path=None):
        """Plot a histogram of energies of each simplex."""
        recs = self.c.find({})
        nRecs = recs.count()
        print "Found %i records" % nRecs
        energies = [rec['best_fopt'] for rec in recs]
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes((0.22,0.15, 0.72,0.8))
        ax.hist(energies, bins=nRecs/10)
        ax.set_xlabel(r"$\mathcal{F}$")
        ax.set_ylabel(r"starts")
        if path is None:
            path = os.path.join(self.workDir, self.cname+"_energies")
        fig.savefig(path+".pdf", bbox='tight')
        fig.savefig(path+".eps", bbox='tight')

    def extime_nrestarts(self, path=None):
        """Plot execution time of a start vs. number of restarts."""
        recs = self.c.find({})
        nRecs = recs.count()
        exTimes = []
        nRestarts = []
        for rec in recs:
            exTimes.append(rec['runtime'])
            nRestarts.append(rec['n_restarts'])

        nullfmt   = NullFormatter()

        # definitions for the axes
        left, width = 0.22, 0.6
        bottom, height = 0.15, 0.65
        bottom_h = bottom+height+0.00
        left_h = left+width+0.00
        hist_width = 1. - left - width - 0.02
        hist_height = 1. - bottom - height - 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, hist_height]
        rect_histy = [left_h, bottom, hist_width, height]

        # start with a rectangular Figure
        fig = plt.figure(figsize=(3,3))

        ax = fig.add_axes(rect_scatter)
        axHistx = fig.add_axes(rect_histx)
        axHisty = fig.add_axes(rect_histy)

        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHistx.yaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        axHisty.xaxis.set_major_formatter(nullfmt)

        # the scatter plot:
        ax.plot(nRestarts, exTimes, 'ok', ms=0.5)
        ax.set_ylabel(r"Execution Time (seconds)")
        ax.set_xlabel(r"$N_{restarts}$")
        
        xmin = min(nRestarts)
        xmax = max(nRestarts)
        ymin = min(exTimes)
        ymax = max(exTimes)
        
        ax.set_xlim((xmin,xmax))
        ax.set_ylim((ymin,ymax))

        axHistx.hist(nRestarts, bins=nRecs/10, histtype='step',
                color='k')
        axHisty.hist(exTimes, bins=nRecs/10, orientation='horizontal',
                histtype='step', color='k')

        axHistx.set_xlim(ax.get_xlim())
        axHisty.set_ylim(ax.get_ylim())
        
        if path is None:
            path = os.path.join(self.workDir, self.cname+"_runtimes")
        fig.savefig(path+".pdf", bbox='tight')
        fig.savefig(path+".eps", bbox='tight')



if __name__ == '__main__':
    main()


