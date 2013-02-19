import os
import subprocess

import montage
import pyfits
import numpy as np

from owl.astromatic import Swarp

import blockdb
import mosaicdb


class MontagePipeline(object):
    """Pipeline wrapper for montage"""
    def __init__(self, rootName, blockSel, band, mosaicName, workDir,
            levelOnly=True):
        super(MontagePipeline, self).__init__()
        self.rootName = rootName
        self.blockSel = blockSel
        self.mosaicName = mosaicName
        self.workDir = workDir
        self.band = band

        self.origDirectory = os.getcwd()
        if levelOnly:
            self.skyType = "scalar"
            self.levelOnly = True
        else:
            self.skyType = "planar"
            self.levelOnly = False
        # workDir = "skyoffsets/montage/%s_%s" % (band,skyType)
        self.projDir = "proj"
        self.diffDir = "diff"
        self.correctedDir = "corr"
        self.rawTblPath = "m31_%s.raw.tbl" % band
        self.projTblPath = "m31_%s.proj.tbl" % band
        self.correctedTblPath = "M31_%s.corr.tbl" % band
        self.hdrPath = "m31_%s.hdr" % band
        self.projStatsPath = "m31_%s.projstats" % band
        self.diffsPath = "m31_%s.diffs" % band
        self.diffFitsPath = "m31_%s.difffits" % band
        self.correctionsPath = "m31_%s.corrections" % band
        self.mosaicPath = "m31_%s_montage_%s.fits" % (band, self.skyType)

    def make_mosaic(self):
        """docstring for make_mosaic"""
        # if not os.path.exists(workDir):
        print "Swarp pre-re-sampling"
        # need to create resampled image set
        print "workDir", self.workDir
        self.resample_blocks(self.blockSel, self.band, self.workDir)

        os.chdir(self.workDir)  # change

        # make metadata table
        if os.path.exists(self.rawTblPath) is False:
            print "mImgtbl"
            print montage.mImgtbl(".", self.rawTblPath, recursive=False, corners=True,
                include_area=False, debug=False, output_invalid=False,
                status_file=None, fieldlist=None, img_list=None)
        # Create output image template
        if os.path.exists(self.hdrPath) is False:
            print "mMakeHdr"
            print montage.mMakeHdr(self.rawTblPath, self.hdrPath, north_aligned=True,
                    system="EQUJ")
        print "mHdrCheck"
        print montage.mHdrCheck(self.hdrPath)  # verify that the WCS is valid
        # reproject images into TAN system
        if os.path.exists(self.projDir) is False:
            os.makedirs(self.projDir)
            print "mProjExec"
            print montage.mProjExec(self.rawTblPath,
                        self.hdrPath, self.projDir,
                        self.projStatsPath,
                        raw_dir=None, debug=False, exact=False, whole=False,
                        border=None,
                        restart_rec=None, status_file=None, scale_column=None,
                        mpi=False,
                        n_proc=8)
        # Create a new image table with Montage's reprojected images
        if os.path.exists(self.projTblPath) is False:
            print montage.mImgtbl(self.projDir, self.projTblPath)
        # Detect overlaps and compute difference images
        if os.path.exists(self.diffsPath) is False:
            print "mOverlaps"
            print montage.mOverlaps(self.projTblPath, self.diffsPath, exact=True,
                    debug_level=None, status_file=None)
            if os.path.exists(self.diffDir) is False: os.makedirs(self.diffDir)
            print "mDiffExec"
            print montage.mDiffExec(self.diffsPath, self.hdrPath, self.diffDir,
                    proj_dir=self.projDir,
                    no_area=False, status_file=None, mpi=False, n_proc=8)
        # Fit the difference images
        if os.path.exists(self.diffFitsPath) is False:
            #print "mDiffFitExec"
            #print montage.mDiffFitExec(diffsPath, diffFitsPath, diffDir,
            #debug=True)
            print "mFitExec"
            print montage.mFitExec(self.diffsPath, self.diffFitsPath, self.diffDir)
        # Model background corrections
        if os.path.exists(self.correctionsPath) is False:
            print "mBgModel"
            print montage.mBgModel(self.projTblPath, self.diffFitsPath, self.correctionsPath,
                    n_iter=32767, level_only=self.levelOnly)
        # Apply background corrections
        if os.path.exists(self.correctedDir) is False:
            os.makedirs(self.correctedDir)
            print montage.mBgExec(self.projTblPath, self.correctionsPath,
                    self.correctedDir,
                    proj_dir=self.projDir)
        # Create a mosaic
        if os.path.exists(self.mosaicPath) is False:
            print montage.mImgtbl(self.correctedDir, self.correctedTblPath)
            print montage.mAdd(self.correctedTblPath, self.hdrPath, self.mosaicPath,
                    img_dir=self.correctedDir, type="median")
        os.chdir(self.origDirectory)
        
        fullMosaicPath = os.path.join(self.workDir, self.mosaicPath)

        tiffPath = os.path.join(self.workDir, self.mosaicName + ".tif")
        subprocess.call("stiff -VERBOSE_TYPE QUIET %s -OUTFILE_NAME %s"
                % (fullMosaicPath, tiffPath), shell=True)

        # Add to the mosaicdb
        mosaicDB = mosaicdb.MosaicDB(dbname="m31", cname="mosaics")
        if mosaicDB.collection.find({"_id": self.mosaicName}).count() > 0:
            mosaicDB.collection.remove({"_id": self.mosaicName})
        doc = {"_id": self.mosaicName, "image_path": fullMosaicPath,
                "subsampled_path": fullMosaicPath,
                "FILTER": self.band,
                "calibrated_set": self.rootName,
                "montage_type": self.skyType}
        mosaicDB.collection.insert(doc)

    def resample_blocks(self, blockSel, band, workDir, pixScale=1.):
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
        configs = {"RESAMPLE_DIR": self.workDir,
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

    def get_bg_corrections(self):
        """Return data with the correction levels or planes"""
        # TODO this is currently only optimized for planes
        imID, a, b, c = np.loadtxt(self.correctionsPath, unpack=True,
                skiprows=1)
        # Match to fieldnames
        imgIDRAW, imgNamesRAW = np.loadtxt(self.rawTblPath, unpack=True,
                skiprows=3, dtype=[('id', int), ('name', str, 35)])
        fieldNames = [name.split('_')[0] for name in imgNamesRAW]
        fieldNamesSorted = [fieldNames[i] for i in imID]
        return a, b, c, fieldNamesSorted

    def get_diffs(self):
        """Return a record array with difference image info, and also return
        lists identifying field names corresponding to indices."""
        # TODO
        # Alternative: model background of the corrected images instead
        # of trying to do the math on plane offset images
        # Yes, do this with the corr/ images
        # 1. read difffits table as record array
        dt = [('plus', int), ('minus', int), ('a', float), ('b', float),
                ('c', float), ('crpix1', float), ('crpix2', float),
                ('xmin', int), ('xmax', int), ('ymin', int),
                ('ymax', int), ('xcenter', float), ('ycenter', float),
                ('npixel', int), ('rms', float), ('boxx', float),
                ('bloxy', float), ('boxwidth', float), ('boxheight', float),
                ('boxang', float)]
        recArray = np.loadtxt(self.diffFitsPath, skiprows=1, dtype=dt)
        
        # 2. read in raw table, make sorted list of fieldnames by id number
        idNum, imgNamesRAW = np.loadtxt(self.rawTblPath, unpack=True,
                skiprows=3, dtype=[('id', int), ('name', str, 35)])
        fieldNames = [name.split('_')[0] for name in imgNamesRAW]
        return recArray, fieldNames


def scale_image(path):
    fits = pyfits.open(path)
    data = fits[0].data
    data[data == 0.] = np.nan
    fits[0].data = data
    fits.writeto(path, clobber=True)
