import os
import shutil
import glob

import montage
import pyfits
import numpy as np

from owl.astromatic import Swarp

import blockdb
import mosaicdb

ALLFIELDS = ["M31-%i" %i for i in  range(1,28) + range(32,44)]

def main():
    #work_montage("J", levelOnly=True)
    work_montage("J", levelOnly=False)
    #work_montage("Ks", levelOnly=True)
    work_montage("Ks", levelOnly=False)


def work_montage(rootName, blockSel, band, mosaicName, workDir,
        levelOnly=True):
    origDirectory = os.getcwd()
    if levelOnly:
        skyType = "scalar"
    else:
        skyType = "planar"
    # workDir = "skyoffsets/montage/%s_%s" % (band,skyType)
    projDir = "proj"
    diffDir = "diff"
    correctedDir = "corr"
    rawTblPath = "m31_%s.raw.tbl" % band
    projTblPath = "m31_%s.proj.tbl" % band
    correctedTblPath = "M31_%s.corr.tbl" % band
    hdrPath = "m31_%s.hdr" % band
    projStatsPath = "m31_%s.projstats" % band
    diffsPath = "m31_%s.diffs" % band
    diffFitsPath = "m31_%s.difffits" % band
    correctionsPath = "m31_%s.corrections" % band
    mosaicPath = "m31_%s_montage_%s.fits" % (band, skyType)
    
    # if not os.path.exists(workDir):
    print "Swarp pre-re-sampling"
    # need to create resampled image set
    print "workDir", workDir
    resample_blocks(blockSel, band, workDir)

    os.chdir(workDir)  # change

    # make metadata table
    if os.path.exists(rawTblPath) is False:
        print "mImgtbl"
        print montage.mImgtbl(".", rawTblPath, recursive=False, corners=True,
            include_area=False, debug=False, output_invalid=False,
            status_file=None, fieldlist=None, img_list=None)
    # Create output image template
    if os.path.exists(hdrPath) is False:
        print "mMakeHdr"
        print montage.mMakeHdr(rawTblPath, hdrPath, north_aligned=True,
                system="EQUJ")
    print "mHdrCheck"
    print montage.mHdrCheck(hdrPath)  # verify that the WCS is valid
    # reproject images into TAN system
    if os.path.exists(projDir) is False:
        os.makedirs(projDir)
        print "mProjExec"
        print montage.mProjExec(rawTblPath,
                    hdrPath, projDir,
                    projStatsPath,
                    raw_dir=None, debug=False, exact=False, whole=False,
                    border=None,
                    restart_rec=None, status_file=None, scale_column=None,
                    mpi=False,
                    n_proc=8)
    # Create a new image table with Montage's reprojected images
    if os.path.exists(projTblPath) is False:
        print montage.mImgtbl(projDir, projTblPath)
    # Detect overlaps and compute difference images
    if os.path.exists(diffsPath) is False:
        print "mOverlaps"
        print montage.mOverlaps(projTblPath, diffsPath, exact=True,
                debug_level=None, status_file=None)
        if os.path.exists(diffDir) is False: os.makedirs(diffDir)
        print "mDiffExec"
        print montage.mDiffExec(diffsPath, hdrPath, diffDir, proj_dir=projDir,
            no_area=False, status_file=None, mpi=False, n_proc=8)
    # Fit the difference images
    if os.path.exists(diffFitsPath) is False:
        #print "mDiffFitExec"
        #print montage.mDiffFitExec(diffsPath, diffFitsPath, diffDir,
        #debug=True)
        print "mFitExec"
        print montage.mFitExec(diffsPath, diffFitsPath, diffDir)
    # Model background corrections
    if os.path.exists(correctionsPath) is False:
        print "mBgModel"
        print montage.mBgModel(projTblPath, diffFitsPath, correctionsPath,
                   n_iter=32767, level_only=levelOnly)
    # Apply background corrections
    if os.path.exists(correctedDir) is False:
        os.makedirs(correctedDir)
        print montage.mBgExec(projTblPath, correctionsPath, correctedDir,
            proj_dir=projDir)
    # Create a mosaic
    if os.path.exists(mosaicPath) is False:
        print montage.mImgtbl(correctedDir, correctedTblPath)
        print montage.mAdd(correctedTblPath, hdrPath, mosaicPath,
                img_dir=correctedDir, type="median")
    os.chdir(origDirectory)

    # Add to the mosaicdb
    fullMosaicPath = os.path.join(workDir, mosaicPath)
    mosaicDB = mosaicdb.MosaicDB(dbname="m31", cname="mosaics")
    if mosaicDB.collection.find({"_id": mosaicName}).count() > 0:
        mosaicDB.collection.remove({"_id": mosaicName})
    doc = {"_id": mosaicName, "image_path": fullMosaicPath,
            "subsampled_path": fullMosaicPath,
            "FILTER": band,
            "calibrated_set": rootName,
            "montage_type": skyType}
    mosaicDB.collection.insert(doc)


def resample_blocks(blockSel, band, workDir, pixScale=1.):
    """Resamples blocks into the TAN system at a reduced pixel scale,
    compatible with Montage."""
    if not os.path.exists(workDir):
        print "Make dir", workDir
        os.makedirs(workDir)
    # blockDB = blockdb.BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    blockDB = blockdb.BlockDB()
    blockDocs = blockDB.find_blocks(blockSel)
    
    ids, imagePaths, weightPaths = [], [], []
    for blockName, doc in blockDocs.iteritems():
        ids.append(doc['_id'])
        print doc['_id']
        imagePaths.append(doc['image_path'])
        weightPaths.append(doc['weight_path'])
    configs = {"RESAMPLE_DIR": workDir,
            "PROJECTION_TYPE": "TAN",
            "COMBINE": "N", "RESAMPLE": "Y", "WEIGHT_TYPE": "MAP_WEIGHT",
            "NTHREADS": "8", "SUBTRACT_BACK": "N",
            "WRITE_XML": "N", "VMEM_DIR": "/Volumes/Zaphod/tmp",
            "MEM_MAX": "8000",
            'PIXEL_SCALE': "%.2f" % pixScale,
            'PIXELSCALE_TYPE': 'MANUAL'}
    swarp = Swarp(imagePaths, "resamp_%s" % band + ".fits",
            weightPaths=weightPaths, configs=configs,
            workDir=workDir)
    swarp.run()

    for dbid, imagePath in zip(ids, imagePaths):
        basename = os.path.splitext(os.path.basename(imagePath))[0]
        resamplePath = os.path.join(workDir, basename + ".resamp.fits")
        resampledWeightPath = os.path.join(workDir,
                basename + ".resamp.weight.fits")
        os.remove(resampledWeightPath)  # don't need weights
        scale_image(resamplePath)
        blockDB.collection.update({"_id": dbid},
                {"$set": {"montage_input_image_path": resamplePath}})

    # Clean up Swarp's input files
    os.remove(os.path.join(workDir, "defaults.txt"))
    os.remove(os.path.join(workDir, "inputlist.txt"))
    os.remove(os.path.join(workDir, "weightlist.txt"))


def scale_image(path):
    fits = pyfits.open(path)
    data = fits[0].data
    data[data == 0.] = np.nan
    fits[0].data = data
    fits.writeto(path, clobber=True)

if __name__ == '__main__':
    main()
