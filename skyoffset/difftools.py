"""Functions and classes that allow images to be differenced"""
import os
import cPickle
import numpy as np
import scipy.stats
import astropy.io.fits
from astropy.stats.funcs import sigma_clip
import Polygon
import Polygon.Utils
import multiprocessing

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class SliceableBase(object):
    """Baseclass for dealing with image slices."""

    def _apply_mask(self, image, mask):
        """Set masked pixels to NaN."""
        image[mask > 0] = np.nan
        return image
    
    def getGoodPix(self, fracThreshold=0.5, threshold=None):
        if threshold is None:
            threshold = fracThreshold * (self.weight.max()
                    - self.weight.min()) + self.weight.min()
        return self.weight > threshold
    
    def saveSegments(self, pathRoot):
        """Saves a the sliced FITS image to pathRoot_im.fits and the sliced
        weight FITS to pathRoot_wht.fits."""
        dirname = os.path.dirname(pathRoot)
        if os.path.exists(dirname) is False:
            os.makedirs(dirname)
        astropy.io.fits.writeto(pathRoot + "_im.fits", self.image,
                clobber=True)
        if self.weight is not None:
            astropy.io.fits.writeto(pathRoot + "_wht.fits", self.weight,
                    clobber=True)


class SliceableImage(SliceableBase):
    """Used by _computeDiff to efficiently slice overlapping images and compare
    their common set of good pixels.
    """
    def __init__(self, key, image, weight, mask=None):
        super(SliceableImage, self).__init__()
        self.key = key
        self.fullImage = image
        self.fullWeight = weight
        if mask is not None:
            self.fullImage = self._apply_mask(image, mask)
            self.fullWeight = self._apply_mask(weight, mask)
        self.r = None
        self.image = None
        self.weight = None
    
    @classmethod
    def makeFromFITS(cls, key, imagePath, weightPath, mask_path=None):
        """Create a `SliceableImage` from FITS paths"""
        print imagePath, weightPath, mask_path
        if weightPath is not None:
            w = astropy.io.fits.getdata(weightPath, 0)
        else:
            w = None
        if mask_path is not None:
            m = astropy.io.fits.getdata(mask_path, 0)
        else:
            m = None
        return cls(key, astropy.io.fits.getdata(imagePath, 0), w, mask=m)
    
    def _slice(self, im):
        return im[self.r[1][0]:self.r[1][1], self.r[0][0]:self.r[0][1]]
    
    def setRange(self, sliceRange):
        """docstring for setRange"""
        self.r = sliceRange
        self.image = self._slice(self.fullImage)
        if self.fullWeight is not None:
            self.weight = self._slice(self.fullWeight)
        else:
            self.weight = None


class SliceableFITS(SliceableBase):
    """Used for accessing slices of a large FITS image. A semi-replacement for
    using the Sliceableimage.makeFromFITS interface. This class will read the
    image slice in memory as the range is set, thus minimizing memory usage.
    """
    def __init__(self, key, imagePath, weightPath):
        super(SliceableFITS, self).__init__()
        self.key = key
        self.imagePath = imagePath
        self.weightPath = weightPath
        self.imageFITS = astropy.io.fits.open(self.imagePath)
        self.weightFITS = astropy.io.fits.open(self.weightPath)
        self.image = None
        self.weight = None
        self.r = None
    
    def setRange(self, sliceRange):
        """docstring for setRange"""
        self.r = sliceRange
        imageSec = self.imageFITS[0].section[self.r[1][0]:self.r[1][1], :]
        weightSec = self.weightFITS[0].section[self.r[1][0]:self.r[1][1], :]
        self.image = imageSec[:, self.r[0][0]:self.r[0][1]]
        self.weight = weightSec[:, self.r[0][0]:self.r[0][1]]
    
    def close(self):
        """Deallocate the FITS file references."""
        self.imageFITS.close()
        self.weightFITS.close()


class ResampledWCS(object):
    """Holds the WCS of a single resampled image frame.
    
    .. TODO:: rename to be unlike `frame`.
    """
    def __init__(self, header):
        super(ResampledWCS, self).__init__()
        self.header = header
        self.naxis1 = self.header['NAXIS1']
        self.naxis2 = self.header['NAXIS2']
        self.crpix1 = int(self.header['crpix1']) - 1  # to be 0-based
        self.crpix2 = int(self.header['crpix2']) - 1
        self.makePolygon()
    
    def makePolygon(self):
        """Makes self.polygon, which is a `Polygon` instance with vertices in
        the primed space of the mosaic image.
        """
        polyPoints = self.getVertices()
        self.polygon = Polygon.Polygon(polyPoints)
    
    def getVertices(self):
        """docstring for getVertices"""
        print self.naxis1, self.naxis2, self.crpix1, self.crpix2
        vertsX = np.array([0, 0, self.naxis1 - 1, self.naxis1 - 1],
                dtype=int) - self.crpix1
        vertsY = np.array([0, self.naxis2 - 1, self.naxis2 - 1, 0],
                dtype=int) - self.crpix2
        vertsX = list(vertsX)
        vertsY = list(vertsY)
        polyPoints = zip(vertsX, vertsY)
        return polyPoints
    
    def getCentroid(self):
        """:return: (x,y) tuple of the frame centroid in the mosaic space."""
        x = np.array([0, 0, self.naxis1 - 1, self.naxis1 - 1], dtype=int) \
                - self.crpix1
        y = np.array([0, self.naxis2 - 1, self.naxis2 - 1, 0], dtype=int) \
                - self.crpix2
        return x.mean(), y.mean()


class ManualResampledWCS(ResampledWCS):
    """Manually create a ResampledWCS from NAXISi and CRPIXi."""
    def __init__(self, naxis1, naxis2, crpix1, crpix2):
        """docstring for __init__"""
        self.naxis1 = naxis1
        self.naxis2 = naxis2
        self.crpix1 = int(crpix1) - 1  # so that everything is zero-based
        self.crpix2 = int(crpix2) - 1
        self.makePolygon()


class Overlap(object):
    """For an overlapping pair of images, records the slice of their overlap.
    """
    def __init__(self, upperFrame, lowerFrame):
        super(Overlap, self).__init__()
        self.upperFrame = upperFrame
        self.lowerFrame = lowerFrame
        
        self._computeOverlap()
        
        if self.overlap is not None:
            self.upSlice = self._makeOverlapRangeForImageFrame(self.upperFrame)
            self.loSlice = self._makeOverlapRangeForImageFrame(self.lowerFrame)
    
    def _computeOverlap(self):
        """Determins the polygon of the overlap between the two frames."""
        if self.upperFrame.polygon.overlaps(self.lowerFrame.polygon):
            self.overlap = self.upperFrame.polygon & self.lowerFrame.polygon
        else:
            self.overlap = None
    
    def _makeOverlapRangeForImageFrame(self, frame):
        """Converts a polygon in co-add space to a range across the image's
        axes.
        """
        # Convert the overlap polygon (in the prime co-add space) to the
        # coordinates of the resampled image
        primeVerts = Polygon.Utils.pointList(self.overlap)
        xlist = []
        ylist = []
        for (xPrime, yPrime) in primeVerts:
            xlist.append(xPrime + frame.crpix1)
            ylist.append(yPrime + frame.crpix2)
        ymin = int(min(ylist))
        ymax = int(max(ylist))
        xmin = int(min(xlist))
        xmax = int(max(xlist))
        # add one to the upper limit because of the behaviour of range() func
        return (xmin, xmax + 1), (ymin, ymax + 1)
    
    def hasOverlap(self):
        """docstring for hasOverlap"""
        if self.overlap is None:
            return False
        else:
            return True
    
    def getUpperLowerSlices(self):
        """:return: tuple (upper slice, lower slice)"""
        return self.upSlice, self.loSlice
    
    def getOverlapCRPIX(self, r, frame):
        """:return: tuple (CRPIX1, CRPIX2) of the overlapping image frame."""
        # The left slice of the upper image. This is the amount that the CRPIX1
        # will shift by.
        deltaX = r[0][0]
        crpix1 = frame.crpix1 - deltaX
        # Do the same with the yaxis
        deltaY = r[1][0]
        crpix2 = frame.crpix2 - deltaY
        return (crpix1, crpix2)
    
    def getLowerOverlapCRPIX(self):
        """docstring for getLowerOverlapCRPIX"""
        return self.getOverlapCRPIX(self.loSlice, self.lowerFrame)
    
    def getUpperOverlapCRPIX(self):
        """docstring for getLowerOverlapCRPIX"""
        return self.getOverlapCRPIX(self.upSlice, self.upperFrame)
    
    def getUpperCentreTrans(self):
        """:return: (delta x, delta y) tuple of offsets to go from the centre
        of the upper frame, to the centre of the overlap frame."""
        return self._makeCentreTrans(self.upperFrame, self.upSlice)
    
    def getLowerCentreTrans(self):
        """:return: (delta x, delta y) tuple of offsets to go from the centre
        of the upper frame, to the centre of the overlap frame."""
        return self._makeCentreTrans(self.lowerFrame, self.loSlice)
    
    def _makeCentreTrans(self, frame, slce):
        oXMin, oXMax = slce[0]
        oYMin, oYMax = slce[1]
        oWidth = oXMax - oXMin
        oHeight = oYMax - oYMin
        
        fWidth = frame.naxis1
        fHeight = frame.naxis2
        
        xDelta = oXMin + oWidth / 2. - fWidth / 2.
        yDelta = oYMin + oHeight / 2. - fHeight / 2.
        return xDelta, yDelta


class OverlapDB(object):
    """Container for finding and holding all overlaps between a set of
    ResampledWCS.
    """
    def __init__(self, resampledWCSs):
        super(OverlapDB, self).__init__()
        # frame/field dictionary of ResampledWCS instances
        self.resampledWCSs = resampledWCSs
        self._findOverlaps()
    
    def __getitem__(self, targetField):
        """:return: a list of all fields that have overlaps with `targetField`.
        """
        coupledFields = []
        for overlapKey in self.overlaps.keys():
            if targetField in overlapKey:
                if targetField == overlapKey[0]:
                    coupledFields.append(overlapKey[1])
                else:
                    coupledFields.append(overlapKey[0])
        return list(set(coupledFields))
    
    def _findOverlaps(self):
        """docstring for findOverlaps"""
        self.overlaps = {}
        frameIDs = self.resampledWCSs.keys()
        for i, frameID1 in enumerate(frameIDs):
            for frameID2 in frameIDs[i + 1:]:
                overlap = Overlap(self.resampledWCSs[frameID1],
                        self.resampledWCSs[frameID2])
                if overlap.hasOverlap():
                    overlapKey = (frameID1, frameID2)
                    self.overlaps[overlapKey] = overlap
    
    def iteritems(self):
        """Delegate iteritems to the overlaps dictionary"""
        return self.overlaps.iteritems()


class Couplings(object):
    """Computes and stores image differences for overlapping fields."""
    def __init__(self):
        super(Couplings, self).__init__()
        self.fields = {}
        self.fieldDiffs = {}
        self.fieldDiffSigmas = {}
        self.fieldDiffAreas = {}
        self.fieldLevels = {}
        self.coverage_fractions = {}
        self.diff_paths = {}
    
    def __getitem__(self, fieldname):
        """:return: list of fields that are coupled to the named field."""
        return self.overlapDB[fieldname]
    
    def get_doc(self):
        """Makes a dictionary document for saving the couplings in MongoDB.
        
        The field tuples are replaced with a single string key using '*'
        as the delimter.
        """
        diffs = {}
        sigmas = {}
        areas = {}
        levels = {}
        coverage_fractions = {}
        diff_paths = {}
        for pair, diff in self.fieldDiffs.iteritems():
            diffs["*".join(pair)] = float(diff)
        for pair, sigma in self.fieldDiffSigmas.iteritems():
            sigmas["*".join(pair)] = float(sigma)
        for pair, area in self.fieldDiffAreas.iteritems():
            areas["*".join(pair)] = float(area)
        for pair, level in self.fieldLevels.iteritems():
            levels["*".join(pair)] = float(level)
        for pair, f in self.coverage_fractions.iteritems():
            coverage_fractions["*".join(pair)] = float(f)
        for pair, p in self.diff_paths.iteritems():
            diff_paths["*".join(pair)] = p
        doc = {"fields": self.fields,
            "diffs": diffs,
            "sigmas": sigmas,
            "areas": areas,
            "levels": levels,
            "coverage_fractions": coverage_fractions,
            "diff_paths": diff_paths}
        return doc

    @classmethod
    def load_doc(cls, doc):
        """Make a Couplings instance from a MongoDB document.
        
        The mongoDB document simulates tuples of field pairs as a string with
        '*' as the delimeter.
        """
        instance = cls()
        instance.fields = doc['fields']
        instance.fieldDiffs = {}
        instance.fieldDiffSigmas = {}
        instance.fieldDiffAreas = {}
        instance.fieldLevels = {}
        instance.coverage_fractions = {}
        instance.diff_paths = {}
        for fieldPair, subDoc in doc['diffs'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.fieldDiffs[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['sigmas'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.fieldDiffSigmas[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['areas'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.fieldDiffAreas[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['levels'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.fieldLevels[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['coverage_fractions'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.coverage_fractions[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['diff_paths'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.diff_paths[fieldTuple] = subDoc
        return instance

    def save(self, path):
        """Pickle the couplings object to the given `path`."""
        dirname = os.path.dirname(path)
        if os.path.exists(dirname) is False:
            os.makedirs(dirname)
        if os.path.exists(path):
            os.remove(path)
        
        f = open(path, 'w')
        p = cPickle.Pickler(f)
        p.dump(self.fieldDiffs)
        p.dump(self.fieldDiffSigmas)
        p.dump(self.fieldDiffAreas)
        p.dump(self.fieldLevels)
        p.dump(self.coverage_fractions)
        p.dump(self.diff_paths)
        f.close()
    
    @classmethod
    def load(cls, path, fields):
        """Load a Couplings object from a pickle at the given `path`. Fields
        is a dictionary of fieldname: field object. Only couplings involving
        these particular fields will be used.
        """
        f = open(path, 'r')
        p = cPickle.Unpickler(f)
        instance = cls()
        instance.fields = fields
        instance.fieldDiffs = p.load()
        instance.fieldDiffSigmas = p.load()
        instance.fieldDiffAreas = p.load()
        instance.fieldLevels = p.load()
        instance.coverage_fractions = p.load()
        instance.diff_paths = p.load()
        f.close()
        
        instance._trim_diffs_to_fields()
        return instance
    
    def _trim_diffs_to_fields(self):
        """When loading a couplings object from a pickle, you may want to
        use fewer field objects than went into the original couplings DB.
        """
        fieldNames = self.fields.keys()
        pairKeys = self.fieldDiffs.keys()
        for pairKey in pairKeys:
            field1, field2 = pairKey
            if (field1 not in fieldNames) or (field2 not in fieldNames):
                del self.fieldDiffs[pairKey]
                del self.fieldDiffSigmas[pairKey]
                del self.fieldDiffAreas[pairKey]
                del self.fieldLevels[pairKey]
                del self.coverage_fractions[pairKey]
                del self.diff_paths[pairKey]
    
    def get_overlap_db(self):
        return self.overlapDB

    def add_field(self, name, imagePath, weightPath, mask_path=None):
        """Adds a field object, which is any object that responds to the field
        specification...
        """
        self.fields[name] = {'image_path': imagePath,
                'weight_path': weightPath,
                'mask_path': mask_path}
    
    def make(self, diffDir, mp=True, plotDir=None):
        """Compute coupled image differences, using `diffDir` as a working
        directory for the image differencing.

        Parameters
        ----------
        diffDir : str
            Directory where difference images are saved.
        mp : bool
            If True, then difference images are computed in parallel.
        """
        self._find_overlaps()
        self._compute_overlap_diffs(diffDir, mp=mp, plotDir=plotDir)

    def _find_overlaps(self):
        """Given the fields, detects and records all overlaps between fields.
        """
        fieldFootprints = {}
        for fieldname, field in self.fields.iteritems():
            fieldFootprints[fieldname] = self._make_imageframe(
                field['image_path'])
        self.overlapDB = OverlapDB(fieldFootprints)

    def _make_imageframe(self, imagePath):
        """:return: the ResampledWCS corresponding to image (assumed in ext 0).
        """
        resampledWCS = ResampledWCS(astropy.io.fits.getheader(imagePath, 0))
        return resampledWCS
        
    def _compute_overlap_diffs(self, diffImageDir, mp=True, plotDir=None):
        """Compute the scalar offsets between fields in each overlap.
        
        This is done with multiprocessing, so that overlaps are computed in
        parallel.
        
        :param diffImageDir: can optionally be set to a directory name so
            that the computed difference images will be saved there.
        """
        if plotDir is None:
            plotDir = diffImageDir
        print "Plot Dir is", plotDir
        if os.path.exists(diffImageDir) is False: os.makedirs(diffImageDir)
        if os.path.exists(plotDir) is False: os.makedirs(plotDir)
        
        args = []
        for overlapKey, overlap in self.overlapDB.iteritems():
            field1, field2 = overlapKey
            field1Path = self.fields[field1]['image_path']
            field1WeightPath = self.fields[field1]['weight_path']
            field1MaskPath = self.fields[field1]['mask_path']
            field2Path = self.fields[field2]['image_path']
            field2WeightPath = self.fields[field2]['weight_path']
            field2MaskPath = self.fields[field2]['mask_path']
            arg = (field1, field1Path, field1WeightPath, field1MaskPath,
                    field2, field2Path, field2WeightPath, field2MaskPath,
                    overlap, diffImageDir, plotDir, 5)
            args.append(arg)
        if mp:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            results = pool.map(_computeDiff, args)
            pool.close()
            pool.join()
        else:
            results = map(_computeDiff, args)
        diffs = {}
        diffSigmas = {}
        diffAreas = {}
        meanLevels = {}
        coverage_fractions = {}
        diff_paths = {}
        for result in results:
            field1, field2, offsetData = result
            pairKey = (field1, field2)
            if offsetData is not None:
                diffs[pairKey] = offsetData['mean']
                diffSigmas[pairKey] = offsetData['sigma']
                diffAreas[pairKey] = offsetData['area']
                meanLevels[pairKey] = offsetData['level']
                coverage_fractions[pairKey] = offsetData['coverage_frac']
                diff_paths[pairKey] = offsetData['diff_path']
            else:
                # self.overlaps.rejectOverlap(pairKey)
                pass
        self.fieldDiffs = diffs
        self.fieldDiffSigmas = diffSigmas
        self.fieldDiffAreas = diffAreas
        self.fieldLevels = meanLevels
        self.coverage_fractions = coverage_fractions
        self.diff_paths = diff_paths
    
    def get_field_diffs(self, omittedFields=[], replicatedFields=[]):
        """Returns a dictionary of field differences, and a list of fields
        represented.
        
        Parameters
        ----------
        omittedFields : list
            A list of fields whose couplings should be omitted
            from the list of field dfferences.
        replicatedFields : list
            A list of fields whose couplings should be repeated
            (ie, given more weight.) A field can be replicated an arbitrary
            number of times. This is useful for jackknife/bootstrap analysis.
        """
        # Split the fieldDiffs dictionary into lists of keys and values
        # (this is done because replication will require duplicated keys)
        pairNames = []
        diffs = []
        diffSigmas = {}
        for pairName, diff in self.fieldDiffs.iteritems():
            pairNames.append(pairName)
            diffs.append(diff)
            diffSigmas[pairName] = self.fieldDiffSigmas[pairName]
        print "There are %i pairs initially" % len(pairNames)
        pairNames, diffs = self._getFieldDiffsAfterReplication(pairNames,
            diffs, replicatedFields)
        print "There are %i pairs after replication" % len(pairNames)
        pairNames, diffs = self._getFieldDiffsAfterOmission(pairNames, diffs,
            omittedFields)
        print "There are %i pairs after omission" % len(pairNames)
        print "Should have omitted %i fields" % len(omittedFields)
        # Now get a set of the represented fields
        allFields = []
        for pairKey in pairNames:
            allFields.append(pairKey[0])
            allFields.append(pairKey[1])
        allFields = list(set(allFields))
        return pairNames, diffs, diffSigmas, allFields
    
    def _getFieldDiffsAfterReplication(self, pairNames, diffs,
            replicatedFields):
        """Duplicates items in pairNames and diffs for each match item in
        replicatedFields.
        """
        extraPairNames = []
        extraDiffs = []
        for replicatedField in replicatedFields:
            for i, pairName in enumerate(pairNames):
                if replicatedField in pairName:
                    # duplicate this coupling
                    extraPairNames.append(pairNames[i])
                    extraDiffs.append(diffs[i])
        pairNames += extraPairNames
        diffs += extraDiffs
        return pairNames, diffs
    
    def _getFieldDiffsAfterOmission(self, pairNames, diffs, omittedFields):
        """Omits an coupling that includes the omitted fields."""
        retainedPairNames = []
        retainedDiffs = []
        for i, pairName in enumerate(pairNames):
            keep = True
            for field in pairName:
                if field in omittedFields:
                    keep = False
            if keep == True:
                retainedPairNames.append(pairNames[i])
                retainedDiffs.append(diffs[i])
        return retainedPairNames, retainedDiffs


def _computeDiff(arg):
    """Worker: Computes the DC offset of frame-coadd"""
    upperKey, upperPath, upperWeightPath, upperMaskKey, lowerKey, lowerPath, \
            lowerWeightPath, lowerMaskKey, \
            overlap, diffDir, diffPlotDir, nsigma = arg
    upper = SliceableImage.makeFromFITS(upperKey, upperPath, upperWeightPath,
            mask_path=upperMaskKey)
    upper.setRange(overlap.upSlice)
    lower = SliceableImage.makeFromFITS(lowerKey, lowerPath, lowerWeightPath,
            mask_path=lowerMaskKey)
    lower.setRange(overlap.loSlice)
    goodPix = np.where((upper.weight > 0.) & (lower.weight > 0.))
    nPixels = len(goodPix[0])
    print "diff pair", lowerKey, upperKey, nPixels
    if nPixels > 10:
        # Offset via difference image
        diff_pixels = upper.image[goodPix] - lower.image[goodPix]
        # Choose iters=None so clipping until convergence.
        clipped = sigma_clip(diff_pixels, sig=nsigma, iters=1,
                varfunc=np.nanvar)
        median = np.nanmedian(clipped[~clipped.mask])
        sigma = np.nanstd(clipped[~clipped.mask])
        n_clipped_pixels = len(clipped[~clipped.mask])
        cov_frac = float(n_clipped_pixels) / len(diff_pixels.ravel())
        print "Offset %.2e +/- %.2e" % (median, sigma)
        offsetData = {"mean": median,
                      "sigma": sigma,
                      "area": n_clipped_pixels,
                      "level": median,
                      "coverage_frac": cov_frac}
        
        # Save the difference image, if possible
        if diffDir is not None:
            image = upper.image - lower.image
            print "image median", np.median(image)
            bad_pix = np.where((upper.weight == 0.) | (lower.weight == 0.))
            image[bad_pix] = np.nan
            lower_lim = median - nsigma * sigma
            upper_lim = median + nsigma * sigma
            print "lower_lim, upper_lim", lower_lim, upper_lim
            image[image < lower_lim] = np.nan
            image[image > upper_lim] = np.nan
            path = os.path.join(diffDir, "%s*%s.fits" % (upperKey, lowerKey))
            astropy.io.fits.writeto(path, image, clobber=True)
            offsetData['diff_path'] = path
    else:
        offsetData = None
        path = None
    return upperKey, lowerKey, offsetData


class CoupledPlanes(object):
    """A replacement for the `Couplings` class in the context of slope
    analysis.
    """
    def __init__(self):
        super(CoupledPlanes, self).__init__()
        self.fields = {}
        self.diffPlanes = {}
        self.diffAreas = {}
        self.diffSigmas = {}
        # mean DC level in the overlap
        self.fieldLevels = {}
        # displacement in (x,y) pixels from the centre of the upper image to
        # the overlap centre
        self.upperOverlapTrans = {}
        # same, for the lower image.
        self.lowerOverlapTrans = {}
        self.overlapShape = {}  # (xsize, ysize) of the overlap area
    
    def __getitem__(self, fieldname):
        """:return: a list of fields that are coupled to the same named field.
        """
        return self.overlapsDB[fieldname]

    def get_doc(self):
        """Makes a dictionary document for saving the couplings in MongoDB.
        
        The field tuples are replaced with a single string key using '*'
        as the delimter.
        """
        diffs = {}
        sigmas = {}
        areas = {}
        levels = {}
        upperTrans = {}
        lowerTrans = {}
        overlapShape = {}
        for pair, diff in self.diffPlanes.iteritems():
            diffs["*".join(pair)] = diff
        for pair, sigma in self.diffSigmas.iteritems():
            sigmas["*".join(pair)] = sigma
        for pair, area in self.diffAreas.iteritems():
            areas["*".join(pair)] = area
        for pair, level in self.fieldLevels.iteritems():
            levels["*".join(pair)] = level
        for pair, trans in self.upperOverlapTrans.iteritems():
            upperTrans["*".join(pair)] = trans
        for pair, trans in self.lowerOverlapTrans.iteritems():
            lowerTrans["*".join(pair)] = trans
        for pair, shape in self.overlapShape.iteritems():
            overlapShape["*".join(pair)] = shape

        doc = {"fields": self.fields,
            "diffs": diffs,
            "sigmas": sigmas,
            "areas": areas,
            "levels": levels,
            "upper_trans": upperTrans,
            "lower_trans": lowerTrans,
            "shape": overlapShape}
        return doc

    @classmethod
    def load_doc(cls, doc):
        """Make a CoupledPlanes instance from a MongoDB document.
        
        The mongoDB document simulates tuples of field pairs as a string with
        '*' as the delimeter.
        """
        instance = cls()
        print "loading CoupledPlanes doc", doc
        instance.fields = doc['fields']
        instance.diffPlanes = {}
        instance.diffSigmas = {}
        instance.diffAreas = {}
        instance.fieldLevels = {}
        instance.upperOverlapTrans = {}
        instance.lowerOverlapTrans = {}
        instance.overlapShape = {}
        for fieldPair, subDoc in doc['diffs'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.diffPlanes[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['sigmas'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.diffSigmas[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['areas'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.diffAreas[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['levels'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.fieldLevels[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['upper_trans'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.upperOverlapTrans[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['lower_trans'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.lowerOverlapTrans[fieldTuple] = subDoc
        for fieldPair, subDoc in doc['shape'].iteritems():
            fieldTuple = tuple(fieldPair.split("*"))
            instance.overlapShape[fieldTuple] = subDoc

        return instance
    
    def add_field(self, name, imagePath, weightPath):
        """Adds a field object, which is any object that responds to the field
        specification..."""
        self.fields[name] = {'image_path': imagePath,
            'weight_path': weightPath}
    
    def make(self, diffDir, nThreads=None):
        """Compute coupled image differences, using `diffDir` as a working
        directory for the image differencing.
        """
        self._find_overlaps()
        self._compute_overlap_diffs(diffDir, nProcesses=nThreads)
    
    def _find_overlaps(self):
        """Given the fields, detects and records all overlaps between fields.
        """
        fieldFootprints = {}
        for fieldname, field in self.fields.iteritems():
            fieldFootprints[fieldname] = self._make_imageframe(
                field['image_path'])
        self.overlapDB = OverlapDB(fieldFootprints)

    def _make_imageframe(self, imagePath):
        """:return: the ResampledWCS corresponding to image (assumed in ext 0).
        """
        resampledWCS = ResampledWCS(astropy.io.fits.getheader(imagePath, 0))
        return resampledWCS
    
    def get_overlap_db(self):
        return self.overlapDB
    
    def _compute_overlap_diffs(self, diffImageDir, nProcesses=None):
        """Compute the difference planes between fields in each overlap.
        
        Done with multiprocessing, so that overlaps are computed in parallel.
        
        :param diffImageDir: can be set to a directory name so that the
            computed difference images will be saved there.
        """
        if diffImageDir is not None:
            if os.path.exists(diffImageDir) is False: os.makedirs(diffImageDir)
        
        args = []
        for overlapKey, overlap in self.overlapDB.iteritems():
            field1, field2 = overlapKey
            field1Path = self.fields[field1]['image_path']
            field1WeightPath = self.fields[field1]['weight_path']
            field2Path = self.fields[field2]['image_path']
            field2WeightPath = self.fields[field2]['weight_path']
            arg = (field1, field1Path, field1WeightPath, field2, field2Path,
                    field2WeightPath, overlap, diffImageDir)
            args.append(arg)
        
        if nProcesses is None:
            nProcesses = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=nProcesses)
        results = pool.map(_computeDiffPlane, args)
        # results = map(_computeDiffPlane, args)
        
        self.diffPlanes = {}
        self.diffAreas = {}
        self.diffSigmas = {}
        self.fieldLevels = {}
        self.upperOverlapTrans = {}
        self.lowerOverlapTrans = {}
        self.overlapShape = {}
        for result in results:
            field1, field2, diffData = result
            pairKey = (field1, field2)
            if diffData is not None:
                self.diffPlanes[pairKey] = diffData['plane']
                self.diffSigmas[pairKey] = diffData['sigma']
                self.diffAreas[pairKey] = diffData['area']
                self.fieldLevels[pairKey] = diffData['level']
                self.upperOverlapTrans[pairKey] = diffData['upper_trans']
                self.lowerOverlapTrans[pairKey] = diffData['lower_trans']
                self.overlapShape[pairKey] = diffData['shape']


def _computeDiffPlane(arg):
    """Worker for computing the difference plane between two coupled images."""
    upperKey, upperPath, upperWeightPath, lowerKey, lowerPath, \
        lowerWeightPath, overlap, diffDir = arg
    upper = SliceableImage.makeFromFITS(upperKey, upperPath, upperWeightPath)
    upper.setRange(overlap.upSlice)
    lower = SliceableImage.makeFromFITS(lowerKey, lowerPath, lowerWeightPath)
    lower.setRange(overlap.loSlice)
    goodPix = np.where((upper.weight > 0.) & (lower.weight > 0.))
    nPixels = len(goodPix[0])
    
    if nPixels > 10:
        # Iterate several times until convergence is reached; we're really
        # iterating to build a good description of what the good pixels are
        # Decompose the image into a list of pixels, and their indices
        diffImage = upper.image - lower.image
        shape = diffImage.shape
        ysize, xsize = shape
        y0 = int(ysize / 2.)
        x0 = int(xsize / 2.)

        xCoords = []
        yCoords = []
        xIndices = []
        yIndices = []
        # flagPix = []
        for i in xrange(xsize):
            x = i - x0
            for j in xrange(ysize):
                y = j - y0
                xCoords.append(x)
                yCoords.append(y)
                xIndices.append(i)
                yIndices.append(j)
        xCoords = np.array(xCoords)
        yCoords = np.array(yCoords)
        xIndices = np.array(xIndices, dtype=int)
        yIndices = np.array(yIndices, dtype=int)
        
        converged = False
        planeFit, newGoodPix = _fitPlane(diffImage, xCoords, yCoords,
                xIndices, yIndices, goodPix)
        newGoodPix = _combineGoodPixMasks(diffImage.shape, goodPix, newGoodPix)
        nIter = 0
        while converged is False:
            newPlaneFit, newGoodPix = _fitPlane(diffImage, xCoords, yCoords,
                xIndices, yIndices, newGoodPix)
            
            # Need to combine newGoodPix, which is based purely on sigma clip
            # with the original goodPix. The surface brightness can often be
            # less than the difference flux. Thus, a sigma-clipped newGoodPix
            # will often include non-difference pixels
            newGoodPix = _combineGoodPixMasks(diffImage.shape, goodPix,
                newGoodPix)
            
            planePerDiff = (planeFit - newPlaneFit) / planeFit
            planePerDiffSq = np.sum(planePerDiff ** 2.)
            if planePerDiffSq < 1e-8:
                converged = True
                print "Converged on iter %i" % nIter
            planeFit = newPlaneFit
            nIter += 1
            if nIter > 10:
                break
                print "Iteration timeout"
        
        print "Plane", planeFit
        
        # Get metadata of the couplings
        area = len(goodPix[0])
        meanLevel = np.median((upper.image[newGoodPix]
            + lower.image[newGoodPix]) / 2.)
        # standard deviation of the diffence image to plane
        args = diffImage, xCoords, yCoords, xIndices, yIndices, newGoodPix
        sigma = np.sqrt(_planeObjFunc(planeFit, *args) ** 2.).std()
        
        # The translation to go from the centre of the upper frame to
        # the centre of the overlap
        upperTrans = overlap.getUpperCentreTrans()
        lowerTrans = overlap.getLowerCentreTrans()
        
        offsetData = {"plane": tuple(planeFit), "sigma": sigma, "area": area,
            "level": meanLevel, "upper_trans": upperTrans,
            "lower_trans": lowerTrans, "shape": (xsize, ysize)}
        
        # Save the difference image
        outputImage = np.nan * np.zeros(diffImage.shape)
        outputImage[newGoodPix] = diffImage[newGoodPix]
        astropy.io.fits.writeto(os.path.join(diffDir,
            "%s_%s.fits" % (upperKey, lowerKey)),
            outputImage, clobber=True)
    else:
        offsetData = None
    return upperKey, lowerKey, offsetData


def _fitPlane(diffImage, xCoords, yCoords, xIndices, yIndices, goodPix):
    # Fit the image using least squares
    args = diffImage, xCoords, yCoords, xIndices, yIndices, goodPix
    p0 = [0., 0., 0.]  # mx, my, offset
    p, success = scipy.optimize.leastsq(_planeObjFunc, p0, args=args)
    
    # Build new mask from sigma clipping the difference of model to real image
    mx, my, c = p
    modelImage = np.zeros(diffImage.shape)
    modelImage[yIndices, xIndices] = xCoords * mx + yCoords * my + c
    delta = diffImage - modelImage
    sigma = delta.std()
    goodPix = np.where(delta ** 2. < sigma ** 2.)  # clip 1-sigma deviations
    return p, goodPix


def _planeObjFunc(p, *args):
    realImage, xCoords, yCoords, xIndices, yIndices, goodPix = args
    
    mx, my, c = p
    modelImage = np.zeros(realImage.shape)
    modelImage[yIndices, xIndices] = xCoords * mx + yCoords * my + c
    
    delta = realImage - modelImage
    return np.ravel(delta[goodPix])


def _combineGoodPixMasks(shape, goodPix, newGoodPix):
    goodImage = np.zeros(shape, dtype=int)
    goodImage[goodPix] = 1
    
    newGoodImage = np.zeros(shape, dtype=int)
    newGoodImage[newGoodPix] = 1
    
    # Pixels need to be good in *both* masks
    combinedImage = goodImage & newGoodImage
    combinedGoodPix = np.where(combinedImage == 1)
    return combinedGoodPix


def histogramFit(pixels, initMean, initSigma):
    """Fits the offset and its error by fitting a guassian to the histogram."""
    nPixels = pixels.shape[0]
    nBins = int(nPixels / 1000)
    hist, bins = np.histogram(pixels, bins=nBins)
    binCentres = _makeBinMiddles(bins)
    fitSigma, fitPeak, fitMean = _fitBinnedGaussian(binCentres, hist, initMean,
            initSigma)
    fitSigma = np.sqrt(fitSigma ** 2.)
    return fitPeak, fitSigma, fitMean


def _makeBinMiddles(bins):
    middles = (bins[0:-1] + bins[1:]) / 2.
    return middles


unit_gauss = lambda x, mean, sigma: np.exp(-(mean - x) ** 2
        / (2. * sigma ** 2.))


def _fitBinnedGaussian(bins, counts, initMean, initSigma):
    """Fits a Gaussian directly to a histogram, not the moments of the
    original observations.
    """
    initialGuess = np.array([initSigma, counts.max(), initMean],
            dtype=float)
    fit = scipy.optimize.fmin(_gaussianObjective, initialGuess,
            args=(bins, counts))
    return fit


def _gaussianObjective(fitParams, bins, counts):
    sigma = fitParams[0]
    peak = fitParams[1]
    mean = fitParams[2]
    
    fittedCounts = peak * unit_gauss(bins, mean, sigma)
    diff = np.sum((fittedCounts - counts) ** 2.)
    return diff


def _diffHist(pixels, mean, clippedMedian, sigma, upperKey, lowerKey,
        plotPath):
    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvas(fig)
    fig.subplots_adjust(left=0.15, bottom=0.13, wspace=0.25, hspace=0.25,
        right=0.95)
    ax = fig.add_subplot(111)
    nPixels = len(pixels)
    # Plot histogram
    nCounts, bins, patches = ax.hist(pixels,
        bins=int(nPixels / 1000), histtype='stepfilled', fc='0.5',
        log=True, normed=True)
    # Plot estimates of the image difference
    ax.axvline(mean, ls="-", c='k', label="mean")
    ax.axvline(clippedMedian, ls='--', c='k', label="clipped median")
    # ax.axvline(fittedMean, ls='--', c='r', label="fitted mean")
    ax.legend()
    ax.text(0.05, 0.95,
        r"Clipped $\Delta= %.2e \pm %.2e$" % (clippedMedian, sigma),
        ha="left", va="top", transform=ax.transAxes)
    ax.set_xlabel("Image Difference (counts)")
    upperKey = upperKey.replace("_", "\_")
    lowerKey = lowerKey.replace("_", "\_")
    ax.set_title("%s - %s" % (upperKey, lowerKey))
    canvas.print_figure(plotPath)


class CouplingGraph(object):
    """Allows for couplings to be treated as a directed graph, with trusted
    seeds and descendents.
    
    An example use for this may be that one wants to build an inter-field sky
    offset solution using only a few fields in the core of the galaxy,
    where the S/N is highest. One would start with a seed as the central field,
    and follow the network until a desired number of fields are added.
    
    This code has some similaries to the Couplings/PropagationSolutionNetwork
    classes in DCPropagator.py. The difference is that this class uses
    the `Couplings` object to simply record all overlaps. This class uses that
    in composition, and offers the ability to then iterate through the network,
    from the seed, generation by generation.
    """
    def __init__(self):
        super(CouplingGraph, self).__init__()
        self.pastFieldSet = set([])
    
    def setOverlaps(self, overlapDB):
        """Set a `OverlapDB` object that defines all overlaps between fields.
        """
        self.overlapDB = overlapDB
    
    def setSeed(self, seedname):
        """Sets a single field as the seed."""
        self.seeds = [seedname]
        self.pastFieldSet = set(self.seeds)
    
    def setSeeds(self, seednames):
        """Sets multiple fields as the trusted seed."""
        self.seeds = seednames
        self.pastFieldSet = set(self.seeds)
    
    def __iter__(self):
        return self
    
    def next(self):
        """Iterates through the network, starting from the seed, generation by
        generation.
        """
        # Get the set of fields that are adjacent to fields which are
        # considered 'parents' in the network
        neighboursOfPast = []
        for pastField in self.pastFieldSet:
            neighboursOfPast += self.overlapDB[pastField]
        descendents = set(neighboursOfPast)
        newDescendents = descendents - self.pastFieldSet  # remove prior gens
        if len(newDescendents) == 0:
            raise StopIteration
        self.pastFieldSet = self.pastFieldSet | newDescendents  # update state
        return descendents
