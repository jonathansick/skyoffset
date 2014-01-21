#!/usr/bin/env python
# encoding: utf-8
"""
Make stacks from a set of chip images.

2012-05-18 - Created by Jonathan Sick
"""
import os
import multiprocessing
import astropy.io.fits as pyfits
import numpy

from moastro.astromatic import Swarp

from andpipe.skyoffset.difftools import SliceableImage
from andpipe.skyoffset.difftools import ResampledWCS
from andpipe.skyoffset.difftools import Overlap

from offsettools import apply_offset


def main():
    pass


class ChipStacker(object):
    """General-purpose class for stacking single-extension FITS, adding a
    sky offset to achieve uniform sky bias.
    """
    def __init__(self, stackDB, workDir):
        super(ChipStacker, self).__init__()
        self.workDir = workDir
        if os.path.exists(workDir) is False: os.makedirs(workDir)
        self.stackDB = stackDB

    def pipeline(self, imageKeys, imagePaths, weightPaths, stackName,
            exptimes=None, skyLevels=None, zeropoints=None, stackZP=25.,
            pixScale=None, dbMeta=None, debug=False):
        """Pipeline for running the ChipStacker method to produce stacks
        and addd them to the stack DB.
        """
        cleanCal = False
        if (skyLevels is not None) or (zeropoints is not None):
            imagePaths = self.calibrate_images(imagePaths, exptimes,
                    skyLevels, zeropoints, stackZP, pixScale=pixScale,
                    debug=debug)
            cleanCal = True
        imagePaths = dict(zip(imageKeys, imagePaths))
        weightPaths = dict(zip(imageKeys, weightPaths))
        self.stack_images(imageKeys, imagePaths, weightPaths, stackName)
        self.remove_offset_frames()
        self.renormalize_weight()
        stackDoc = {"_id": stackName,
                "image_path": self.coaddPath,
                "weight_path": self.coaddWeightPath,
                "offsets": self.offsets}
        if dbMeta is not None:
            stackDoc.update(dbMeta)
        self.stackDB.insert(stackDoc)
        if cleanCal:
            for imageKey, imagePath in imagePaths.iteritems():
                os.remove(imagePath)

    def calibrate_images(self, imagePaths, exptimes, skyLevels, zeropoints,
            stackZP, pixScale=None, debug=False):
        """docstring for calibrate_images"""
        calDir = os.path.join(self.workDir, "cal")
        if not os.path.exists(calDir): os.makedirs(calDir)
        args = []
        calImagePaths = []
        for i, path in enumerate(imagePaths):
            if skyLevels is None:
                level = None
            else:
                level = skyLevels[i]
            if zeropoints is None:
                zp = None
            else:
                zp = zeropoints[i]
            calPath = os.path.join(calDir, os.path.basename(path))
            calImagePaths.append(calPath)
            exptime = exptimes[i]
            args.append((path, calPath, exptime, level, zp, stackZP, pixScale))
        if not debug:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            pool.map(_work_image_cal, args)
            pool.terminate()
        else:
            map(_work_image_cal, args)
        return calImagePaths

    def stack_images(self, imageKeys, imagePaths, weightPaths, stackName):
        """Make a stack with the given list of images.
        :param imageKeys: list of strings identifying the listed image paths.
        :param imagePaths: dict of paths to single-extension FITS image files.
        :param weightPaths: dict of paths to weight images for `imagePaths`.
        :return: path to stacked image.
        """
        self.imageKeys = imageKeys
        self.imagePaths = imagePaths
        self.weightPaths = weightPaths
        self.stackName = stackName
        
        # Turn the path lists into dictionaries
        #self.currentOffsetPaths = dict(zip(self.imageKeys, self.imagePaths))
        #self.weightPaths = dict(zip(self.imageKeys, self.weightPaths))
        self.currentOffsetPaths = dict(imagePaths)

        # Make resampled frames
        self.imageFrames = {}
        for imageKey in self.imageKeys:
            header = pyfits.getheader(self.imagePaths[imageKey])
            self.imageFrames[imageKey] = ResampledWCS(header)
        
        # 1. Do initial coadd
        print "Step 1 offset paths", self.currentOffsetPaths
        self.coaddPath, self.coaddWeightPath, self.coaddFrame \
                = self._coadd_frames()
        
        # 2. Compute overlaps of frames to the coadded frame
        self.overlaps = {}
        for imageKey in self.imageKeys:
            self.overlaps[imageKey] \
                    = Overlap(self.imageFrames[imageKey], self.coaddFrame)
        
        # 3. Estimate offsets from the initial mean, and recompute the mean
        diffData = self._compute_differences()
        offsets = self._estimate_offsets(diffData)
        self._make_offset_images(offsets)
        print "Step 3 offset paths", self.currentOffsetPaths
        self.coaddPath, self.coaddWeightPath, self.coaddFrame \
                = self._coadd_frames()
        
        # 4. Recompute offsets to ensure convergence.
        # Use a median for the final stack to get rid of artifacts
        # But need to recompute overlaps to coaddFrame, just in case...
        self.overlaps = {}
        for imageKey in self.imageKeys:
            self.overlaps[imageKey] \
                    = Overlap(self.imageFrames[imageKey], self.coaddFrame)

        diffData = self._compute_differences()
        self.offsets = self._estimate_offsets(diffData)
        self._make_offset_images(self.offsets)
        print "Step 4 offset paths", self.currentOffsetPaths
        self.coaddPath, self.coaddWeightPath, self.coaddFrame \
            = self._coadd_frames(combineType="MEDIAN")

    def remove_offset_frames(self):
        print "currentOffsetPaths"
        print self.currentOffsetPaths
        for imageKey in self.imageKeys:
            os.remove(self.currentOffsetPaths[imageKey])
            print "Delete", self.currentOffsetPaths[imageKey]

    def renormalize_weight(self):
        """Renormalizes the weight image of the stack."""
        fits = pyfits.open(self.coaddWeightPath)
        image = fits[0].data
        image[image > 0.] = 1.
        fits[0].data = image
        fits.writeto(self.coaddWeightPath, clobber=True)

    def _coadd_frames(self, combineType="WEIGHTED"):
        """Swarps images together as their arithmetic mean."""
        imagePathList = []
        weightPathList = []
        for frame in self.currentOffsetPaths:
            imagePathList.append(self.currentOffsetPaths[frame])
            weightPathList.append(self.weightPaths[frame])
        
        configs = {"COMBINE_TYPE": combineType, "WEIGHT_TYPE": "MAP_WEIGHT",
            "PROJECTION_TYPE": "AIT",
            "NTHREADS": "8", "SUBTRACT_BACK": "N",
            "WRITE_XML": "N", "VMEM_DIR": "/Volumes/Zaphod/tmp",
            "MEM_MAX": "8000",
            "COMBINE": "Y",
            "RESAMPLE": "N"}
        swarp = Swarp(imagePathList, self.stackName,
                weightPaths=weightPathList,
                configs=configs, workDir=self.workDir)
        swarp.run()
        coaddPath, coaddWeightPath = swarp.mosaic_paths()
        
        coaddHeader = pyfits.getheader(coaddPath, 0)
        coaddFrame = ResampledWCS(coaddHeader)
        
        return coaddPath, coaddWeightPath, coaddFrame
    
    def _compute_differences(self):
        """Computes the deviation of individual images to the level of the
        average.
        """
        args = []
        for imageKey, overlap in self.overlaps.iteritems():
            # framePath = self.imageLog[imageKey][self.resampledKey][hdu]
            framePath = self.imagePaths[imageKey]
            frameWeightPath = self.weightPaths[imageKey]
            arg = (imageKey, framePath, frameWeightPath, "coadd",
                    self.coaddPath, self.coaddWeightPath, overlap)
            args.append(arg)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = pool.map(_computeDiff, args)
        #results = map(_computeDiff, args)
        offsets = {}
        for result in results:
            frame, coaddKey, offsetData = result
            offsets[frame] = offsetData  # look at _computeDiff() for spec
        pool.terminate()
        return offsets
    
    def _estimate_offsets(self, diffData):
        """Estimate offsets based on the simple difference of taht frame to
        the coadded surface intensity.
        """
        frames = []
        offsets = []
        for frame, data in diffData.iteritems():
            frames.append(frame)
            offsets.append(data['diffimage_mean'])
        offsets = dict(zip(frames, offsets))
        return offsets

    def _make_offset_images(self, offsets):
        """Apply the offsets to the images, and save to disk."""
        if offsets is not None:
            self.currentOffsetPaths = {}
            offsetDir = os.path.join(self.workDir, "offset_frames")
            if os.path.exists(offsetDir) is False:
                os.makedirs(offsetDir)
            
            args = []
            for imageKey in self.imageFrames:
                offset = offsets[imageKey]
                #print "Frame",
                #print imageKey
                #print "offset",
                #print offset
                origPath = self.imagePaths[imageKey]
                offsetPath = os.path.join(offsetDir,
                        os.path.basename(origPath))
                arg = (imageKey, origPath, offset, offsetPath)
                args.append(arg)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            results = pool.map(apply_offset, args)
            #results = map(apply_offset, args)
            for result in results:
                imageKey, offsetImagePath = result
                self.currentOffsetPaths[imageKey] = offsetImagePath
            pool.terminate()


def _work_image_cal(arg):
    """Calibrates images with scalar sky subtraction and zeropoint calibration
    in mag/arcsec^2.
    """
    path, calPath, exptime, level, zp, stackZP, pixScale = arg
    fits = pyfits.open(path)
    if level is not None:
        fits[0].data = fits[0].data - level
    if zp is not None:
        sf = 10. ** (0.4 * (stackZP - zp)) / exptime / (pixScale ** 2.)
        fits[0].data = fits[0].data * sf
        gain = fits[0].header['GAIN']
        gain = gain / sf
        fits[0].header.update("GAIN", gain)
    fits.writeto(calPath, clobber=True)


def _computeDiff(arg):
    """Worker: Computes the DC offset of frame-coadd"""
    upperKey, upperPath, upperWeightPath, lowerKey, lowerPath, \
            lowerWeightPath, overlap = arg
    # print "diff between", upperKey, lowerKey
    upper = SliceableImage.makeFromFITS(upperKey, upperPath, upperWeightPath)
    upper.setRange(overlap.upSlice)
    lower = SliceableImage.makeFromFITS(lowerKey, lowerPath, lowerWeightPath)
    lower.setRange(overlap.loSlice)
    #print "\tUp slice:", overlap.upSlice,
    #print "\tLo slice:", overlap.loSlice
    #print upper.image.shape, lower.image.shape
    #print upper.weight.shape, lower.weight.shape
    goodPix = numpy.where((upper.weight > 0.) & (lower.weight > 0.))
    nPixels = len(goodPix[0])
    if nPixels > 10:
        diffPixels = upper.image[goodPix] - lower.image[goodPix]
        diffPixels = diffPixels[numpy.isfinite(diffPixels)]
        # diffPixelsMean = diffPixels.mean()
        diffPixelsMean = numpy.median(diffPixels)
        diffPixelsSigma = diffPixels.std()
        diffData = {"diffimage_mean": diffPixelsMean,
                    "diffimage_sigma": diffPixelsSigma,
                    "area": nPixels}
    else:
        diffData = None
    return upperKey, lowerKey, diffData

if __name__ == '__main__':
    main()
