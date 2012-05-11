#!/usr/bin/env python
# encoding: utf-8
"""
Functions for making mosaics from blocks given sky offsets.

History
-------
2011-07-25 - Created by Jonathan Sick

"""

__all__ = ['']

import os
import multiprocessing
import pyfits

import owl.astromatic
import offsettools

def block_mosaic(blockDocs, offsets, mosaicName, workDir,
        offset_fcn=offsettools.apply_offset,
        threads=multiprocessing.cpu_count()):
    """Construct a mosaic by offsetting blocks and swarping."""
    offsetDir = os.path.join(workDir, "offset_images")
    if os.path.exists(offsetDir) is False: os.makedirs(offsetDir)

    # apply offsets to block images, then swarp
    args = []
    offsetPaths = {}
    weightPaths = {}
    fieldNames = []
    for stackName, blockDoc in blockDocs.iteritems():
        stackName = blockDoc['_id']
        print "stackName", stackName
        fieldNames.append(stackName)
        imagePath = blockDoc['image_path']
        weightPaths[stackName] = blockDoc['weight_path']
        print "offsets", offsets
        offset = offsets[stackName]
        offsetPath = os.path.join(offsetDir, stackName+".fits")
        offsetPaths[stackName] = offsetPath
        arg = (stackName, imagePath, offset, offsetPath)
        args.append(arg)
    if threads > 1:
        map(offset_fcn, args)
    else:
        pool = multiprocessing.Pool(processes=threads)
        pool.map(offset_fcn, args)
    
    imagePathList = [offsetPaths[k] for k in fieldNames]
    weightPathList = [weightPaths[k] for k in fieldNames]

    # By not resampling, it means that the block will still be in the
    # same resampling system as all other blocks (hopefully)
    swarpConfigs = {"WEIGHT_TYPE": "MAP_WEIGHT",
            "COMBINE_TYPE": "WEIGHTED",
            "RESAMPLE":"N", "COMBINE":"Y"}
    swarp = owl.astromatic.Swarp(imagePathList, mosaicName,
            weightPaths=weightPathList, configs=swarpConfigs,
            workDir=workDir)
    swarp.run()
    blockPath, weightPath = swarp.getMosaicPaths()
    return blockPath, weightPath


def make_block_mosaic_header(blockDocs, mosaicName, workDir):
    """Pre-compute the header of a block mosaic by doing a Swarp dry-run.
    
    Uses Swarp's `HEADER_ONLY=TRUE` mode.
    """
    # apply offsets to block images, then swarp
    offsetPaths = {}
    weightPaths = {}
    fieldNames = []
    for stackName, blockDoc in blockDocs.iteritems():
        stackName = blockDoc['_id']
        fieldNames.append(stackName)
        imagePath = blockDoc['image_path']
        weightPaths[stackName] = blockDoc['weight_path']
        offsetPaths[stackName] = imagePath
    
    imagePathList = [offsetPaths[k] for k in fieldNames]
    weightPathList = [weightPaths[k] for k in fieldNames]

    # By not resampling, it means that the block will still be in the
    # same resampling system as all other blocks (hopefully)
    swarpConfigs = {"WEIGHT_TYPE": "MAP_WEIGHT",
            "COMBINE_TYPE": "WEIGHTED",
            "RESAMPLE":"N", "COMBINE":"N", "HEADER_ONLY":"Y"}
    swarp = owl.astromatic.Swarp(imagePathList, mosaicName,
            weightPaths=weightPathList, configs=swarpConfigs,
            workDir=workDir)
    swarp.run()
    blockPath, weightPath = swarp.getMosaicPaths()
    header = pyfits.getheader(blockPath)
    return header

def subsample_mosaic(fullMosaicPath, pixelScale=1., fluxscale=True):
    """Subsamples the existing mosaic to `pixelScale` (arcsec/pixel)."""
    workDir = os.path.dirname(fullMosaicPath)
    # Make a header of the full mosaic
    configs = {"RESAMPLE_DIR": workDir,
        "COMBINE":"N",
        "RESAMPLE":"Y",
        # "HEADER_ONLY":"Y", # only generate a header image
        # "WEIGHT_TYPE":"MAP_WEIGHT",
        "NTHREADS":"8", "SUBTRACT_BACK":"N",
        "WRITE_XML":"N", "VMEM_DIR":"/Volumes/Zaphod/tmp",
        #"PROJECTION_TYPE": "TAN", # aitoff equal-area projection
        "MEM_MAX": "8000"}
    configs['PIXEL_SCALE'] = "%.2f" % pixelScale
    configs['PIXELSCALE_TYPE'] = 'MANUAL'
    if fluxscale == False:
        configs['FSCALASTRO_TYPE'] = 'NONE'
    swarp = owl.astromatic.Swarp([fullMosaicPath], "downsampled",
    # weightPaths=weightPathList, # not dealing with weight images here
    configs=configs, workDir=workDir)
    swarp.run()
    downsampledPath = os.path.join(workDir,
        os.path.splitext(os.path.basename(fullMosaicPath))[0] + ".resamp.fits")
    return downsampledPath

def test_mosaic_frame():
    """Test functionality of block_mosaic_frame"""
    from blockdb import BlockDB
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")

    blockSel = {"FILTER":"J", "field": {"$nin": ["M31-44","M31-45","M31-46"]}}
    blockDocs = blockDB.find_blocks(blockSel)
    header = make_block_mosaic_header(blockDocs, "test_frame", "skyoffsets/test")
    print header

if __name__ == '__main__':
    test_mosaic_frame()


