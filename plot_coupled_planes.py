#!/usr/bin/env python
# encoding: utf-8
"""
Plots characteristics of the CoupledPlanes in planar mosaics.

History
-------
2011-09-20 - Created by Jonathan Sick

"""

__all__ = ['']

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    pass

def coupling_area_hist(mosaicName, plotPath):
    """docstring for coupling_area_hist"""
    make_plot_dir(plotPath)

    mosaicDB = MosaicDB(dbname="m31", cname="mosaics")
    mosaicDoc = mosaicDB.get_mosaic_doc(mosaicName)
    couplingsDoc = mosaicDoc['couplings']
    
    areas = []
    for pairKey, doc in couplingsDoc.iteritems():
        shape = doc['shape']
        areas.append(shape[0]*shape[1])

def make_plot_dir(path):
    """docstring for make_plot_dir"""
    dirname = os.path.dirname(path)
    if os.path.exists(dirname): os.makedirs(dirname)

if __name__ == '__main__':
    main()


