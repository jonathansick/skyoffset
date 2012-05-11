import pymongo
import os
import multiprocessing
import pyfits
import numpy

from owl.astromatic import Swarp

from wirpipe.disksky.difftools import SliceableImage
from wirpipe.disksky.difftools import ResampledWCS
from wirpipe.disksky.difftools import Overlap

from offsettools import apply_offset

class StackDB(object):
    """Database interface for detector field stacks"""
    def __init__(self, dbname="m31", cname="wircam_stacks", url="localhost",
            port=27017):
        super(StackDB, self).__init__()
        self.dbname = dbname
        self.cname = cname
        self.url = url
        self.port = port
        
        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]

    def stack(self, imageLog, resampledSet, fieldname, band, ext, workDir,
            extraSel={}):
        """Makes a stack for the given set of field-band-extension.
        :param workDir: directory where stacks are made; the stack itself is
            made in workDir/fieldname_band_ext
        """
        sel = {"OBJECT": fieldname, "FILTER": band}
        sel.update(extraSel)
        imageKeys = imageLog.search(sel)
        imagePaths, weightPaths = resampledSet.get_paths(imageKeys, ext)
        frames = imagePaths.keys()
        #frames = ["%s_%i"%(ik,ext) for ik in imageKeys]
        
        stackName = "%s_%s_%i"% (fieldname, band, ext)
        stackDir = os.path.join(workDir, stackName)
        if os.path.exists(stackDir) is False: os.makedirs(stackDir)
        
        print "imagePaths dict", imagePaths

        stacker = ChipStacker(stackDir)
        stacker.stack_images(frames, imagePaths, weightPaths, stackName)
        stacker.remove_offset_frames()
        stacker.renormalize_weight()
        stackPath = stacker.coaddPath
        stackWeightPath = stacker.coaddWeightPath
        # stackFrame = stacker.coaddFrame
        offsets = stacker.offsets # dictionary of scalar offsets for each frame

        # Save the stack to MongoDB
        stackDoc = {"_id": stackName,
                "OBJECT": fieldname,
                "FILTER": band,
                "ext": ext,
                "image_path": stackPath,
                "weight_path": stackWeightPath,
                "offsets": offsets}
        self.collection.insert(stackDoc) # what happens if I upload a duplicate?
    
    def find_stacks(self, sel):
        """Does a MongoDB query for stacks, returning a dictionary (keyed) by
        stack name of the stack documents.
        :param sel: a MongoDB query dictionary.
        """
        recs = self.collection.find(sel)
        docDict = {}
        for r in recs:
            stackName = r['_id']
            docDict[stackName] = r
        return docDict


class ChipStacker(object):
    """General-purpose class for stacking single-extension FITS, adding a
    sky offset to achieve uniform sky bias.
    """
    def __init__(self, workDir):
        super(ChipStacker, self).__init__()
        self.workDir = workDir
        if os.path.exists(workDir) is False: os.makedirs(workDir)

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
        self.coaddPath, self.coaddWeightPath, self.coaddFrame = self._coadd_frames()
        
        # 2. Compute overlaps of frames to the coadded frame
        self.overlaps = {}
        for imageKey in self.imageKeys:
            self.overlaps[imageKey] = Overlap(self.imageFrames[imageKey], self.coaddFrame)
        
        # 3. Estimate offsets from the initial mean, and recompute the mean
        diffData = self._compute_differences()
        offsets = self._estimate_offsets(diffData)
        self._make_offset_images(offsets)
        print "Step 3 offset paths", self.currentOffsetPaths
        self.coaddPath, self.coaddWeightPath, self.coaddFrame = self._coadd_frames()
        
        # 4. Recompute offsets to ensure convergence.
        # Use a median for the final stack to get rid of artifacts
        # But need to recompute overlaps to coaddFrame, just in case...
        self.overlaps = {}
        for imageKey in self.imageKeys:
            self.overlaps[imageKey] = Overlap(self.imageFrames[imageKey], self.coaddFrame)

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

    def _coadd_frames(self, combineType="AVERAGE"):
        """Swarps images together as their arithmetic mean."""
        imagePathList = []
        weightPathList = []
        for frame in self.currentOffsetPaths:
            imagePathList.append(self.currentOffsetPaths[frame])
            weightPathList.append(self.weightPaths[frame])
        
        configs = {"COMBINE_TYPE": combineType, "WEIGHT_TYPE":"MAP_WEIGHT",
            "PROJECTION_TYPE":"AIT",
            "NTHREADS":"8", "SUBTRACT_BACK":"N",
            "WRITE_XML":"N", "VMEM_DIR":"/Volumes/Zaphod/tmp",
            "MEM_MAX": "8000",
            "COMBINE": "Y",
            "RESAMPLE": "N"}
        swarp = Swarp(imagePathList, self.stackName, weightPaths=weightPathList,
            configs=configs, workDir=self.workDir)
        swarp.run()
        coaddPath, coaddWeightPath = swarp.getMosaicPaths()
        
        coaddHeader = pyfits.getheader(coaddPath, 0)
        coaddFrame = ResampledWCS(coaddHeader)
        
        return coaddPath, coaddWeightPath, coaddFrame
    
    def _compute_differences(self):
        """Computes the deviation of individual images to the level of the average."""
        args = []
        for imageKey, overlap in self.overlaps.iteritems():
            # framePath = self.imageLog[imageKey][self.resampledKey][hdu]
            framePath = self.imagePaths[imageKey]
            frameWeightPath = self.weightPaths[imageKey]
            arg = (imageKey, framePath, frameWeightPath, "coadd", self.coaddPath, self.coaddWeightPath, overlap)
            args.append(arg)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = pool.map(_computeDiff, args)
        #results = map(_computeDiff, args)
        offsets = {}
        for result in results:
            frame, coaddKey, offsetData = result
            offsets[frame] = offsetData # look at _computeDiff() for dictionary spec
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
                offsetPath = os.path.join(offsetDir, os.path.basename(origPath))
                arg = (imageKey, origPath, offset, offsetPath)
                args.append(arg)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            results = pool.map(apply_offset, args)
            #results = map(apply_offset, args)
            for result in results:
                imageKey, offsetImagePath = result
                self.currentOffsetPaths[imageKey] = offsetImagePath


def _computeDiff(arg):
    """Worker: Computes the DC offset of frame-coadd"""
    upperKey, upperPath, upperWeightPath, lowerKey, lowerPath, lowerWeightPath, overlap = arg
    # print "diff between", upperKey, lowerKey
    upper = SliceableImage.makeFromFITS(upperKey, upperPath, upperWeightPath)
    upper.setRange(overlap.upSlice)
    lower = SliceableImage.makeFromFITS(lowerKey, lowerPath, lowerWeightPath)
    lower.setRange(overlap.loSlice)
    #print "\tUp slice:", overlap.upSlice,
    #print "\tLo slice:", overlap.loSlice
    #print upper.image.shape, lower.image.shape
    #print upper.weight.shape, lower.weight.shape
    goodPix = numpy.where((upper.weight>0.) & (lower.weight>0.))
    nPixels = len(goodPix[0])
    if nPixels > 10:
        diffPixels = upper.image[goodPix] - lower.image[goodPix]
        diffPixelsMean = diffPixels.mean() # an offset from the pixel-to-pixel difference
        diffPixelsSigma = diffPixels.std()
        diffData = {"diffimage_mean" : diffPixelsMean,
                    "diffimage_sigma": diffPixelsSigma,
                    "area": nPixels}
    else:
        diffData = None
    return upperKey, lowerKey, diffData

        
