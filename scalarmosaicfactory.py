#!/usr/bin/env python
# encoding: utf-8
"""
Make mosaics using scalar sky offsets

2012-05-18 - Created by Jonathan Sick
"""
import os
import numpy as np
import astropy.io.fits
import subprocess

from difftools import Couplings  # Scalar couplings
from andpipe import footprintdb
from multisimplex import SimplexScalarOffsetSolver
import blockmosaic  # common mosaic construction functions
import offsettools
from stackdb import StackDB


class ScalarMosaicFactory(object):
    """docstring for ScalarMosaicFactory"""
    def __init__(self, blockDB, mosaicDB, footprintDB, workDir):
        super(ScalarMosaicFactory, self).__init__()
        self.blockDB = blockDB
        self.mosaicDB = mosaicDB
        self.footprintDB = footprintDB
        self.workDir = workDir
        if not os.path.exists(workDir): os.makedirs(workDir)
    
    def build(self, blockSelector, mosaicName,
            solverDBName, solverCName,
            nRuns=1000, dbMeta=None,
            resetCouplings=False, freshStart=True,
            initScale=5., restartScale=2.):
        """Pipeline facade for building a scalar-sky-offset mosaic"""
        print "Making mosaic", mosaicName
        # Try to load a document with a previous run of this mosaic
        mosaicDoc = self.mosaicDB.get_mosaic_doc(mosaicName)
        if mosaicDoc is None:
            mosaicDoc = {"_id": mosaicName}
            mosaicDoc.update(dbMeta)  # add meta data for this mosaic
            self.mosaicDB.insert(mosaicDoc)
        else:
            print "Found mosaicDoc fields", mosaicDoc.keys()
        
        blockDocs = self.blockDB.find_blocks(blockSelector)
        print "Working on blocks:", blockDocs.keys()
        
        # Make couplings
        if 'couplings' not in mosaicDoc or resetCouplings:
            print "making couplings"
            couplings = self._make_couplings(mosaicName, blockDocs)
        else:
            print "reloading couplings"
            couplings = self._reload_couplings(mosaicDoc['couplings'])
        
        # Precompute the output WCS; add to FootprintDB
        footprintSelector = {"mosaic_name": mosaicName,
            "FILTER": mosaicDoc['FILTER'],
            "kind": "mosaic"}
        self._precompute_mosaic_footprint(blockDocs, self.workDir,
                footprintSelector)

        # Retrieve the ResampledWCS for blocks and mosaic
        footprintDB = footprintdb.FootprintDB()
        mosaicWCS = footprintDB.make_resampled_wcs(footprintSelector)
        
        blockWCSs = {}
        for blockName, blockDoc in blockDocs.iteritems():
            print "the blockDoc:", blockDoc
            field = blockDoc['OBJECT']
            band = blockDoc['FILTER']
            sel = {"field": field, "FILTER": band}
            #blockName = "%s_%s" % (field, band) # Changed!
            print blockName
            blockWCSs[blockName] = footprintDB.make_resampled_wcs(sel)

        self.mosaicDB.collection.update({"_id": mosaicName},
                {"$set": {"solver_cname": mosaicName,
                          "solver_dbname": solverDBName}})

        # Perform optimization
        self._solve_offsets(mosaicName, solverDBName, couplings,
                blockWCSs, mosaicWCS, initScale, restartScale,
                freshStart=freshStart, nRuns=nRuns)

    def _reload_couplings(self, couplingsDocument):
        """Attempt to create a CoupledPlanes instance from a MongoDB
        persisted document."""
        return Couplings.load_doc(couplingsDocument)
    
    def _make_couplings(self, mosaicName, blockDocs):
        """Computes the couplings between blockDocs.
        :return: a difftools.Couplings instance.
        """
        couplings = Couplings()
        for blockName, blockDoc in blockDocs.iteritems():
            print blockDoc
            blockPath = blockDoc['image_path']
            blockWeightPath = blockDoc['weight_path']
            couplings.add_field(blockName, blockPath, blockWeightPath)
        diffImageDir = os.path.join(self.workDir, "diffs")
        couplings.make(diffImageDir)
        couplingsDoc = couplings.get_doc()
        print couplingsDoc
        self.mosaicDB.collection.update({"_id": mosaicName},
                {"$set": {"couplings": couplingsDoc}})
        return couplings

    def _simplex_dispersion(self, initScale, restartScale, couplings):
        """Estimate the standard deviation (about zero offset) to initialize
        the simplex dispersion around.

        Return
        ------
        `initialSigma` and `restartSigma`.
        """
        diffList = [diff for k, diff in couplings.fieldDiffs.iteritems()]
        diffList = np.array(diffList)
        diffSigma = diffList.std()
        return initScale * diffSigma, restartScale * diffSigma
    
    def _solve_offsets(self, mosaicName, solverDBName, couplings,
            blockWCSs, mosaicWCS, initScale, restartScale,
            freshStart=True, nRuns=1000):
        """Use SimplexScalarOffsetSolver to derive offsets for this block."""
        logPath = os.path.join(self.workDir, "%s.log" % mosaicName)
        solver = SimplexScalarOffsetSolver(dbname=solverDBName,
                cname=mosaicName,
                url=self.mosaicDB.url, port=self.mosaicDB.port)

        if freshStart:
            solver.resetdb()
        initSigma, resetSigma = self._simplex_dispersion(initScale,
                restartScale, couplings)
        solver.multi_start(couplings, nRuns, logPath, cython=True, mp=True,
                initSigma=initSigma,
                restartSigma=resetSigma)

        offsets = solver.find_best_offsets()

        self.mosaicDB.collection.update({"_id": mosaicName},
                {"$set": {"offsets": offsets,
                    "solver_cname": mosaicName,
                    "solver_dbname": solverDBName}})
        return solver
    
    def _precompute_mosaic_footprint(self, blockDocs, workDir, metaData):
        """Do a Swarp dry-run to populate the FootprintDB with a record
        of this mosaic.
        :param blockDocs: dictionaries with Block data
        :param workDir: where the Swarp dry-run is performed
        :param metaData: dictionary of key-values that should be saved with
            the footprint in FootprintDB. Its good to declare a mosaic name,
            a kind ("scalar_mosaic",etc), filter and instrument...
        """
        header = blockmosaic.make_block_mosaic_header(blockDocs, "test_frame",
                workDir)
        footprintDB = footprintdb.FootprintDB()
        footprintDB.new_from_header(header, **metaData)

    def make_mosaic(self, mosaicName, blockSel, workDir,
            fieldnames=None, excludeFields=None):
        """Swarp a mosaic using the optimal sky offsets.
        
        The mosaic can be made anytime once entries are added
        to the solver's collection. This is because we initialize
        a SimplexScalarOffsetSolver that re-generates the list of best
        offsets from the collection of solver documents.
        """
        self.workDir = os.path.join(workDir, mosaicName)
        mosaicDoc = self.mosaicDB.collection.find_one({"_id": mosaicName})
        solverCName = mosaicDoc['solver_cname']
        solverDBName = mosaicDoc['solver_dbname']

        if fieldnames is not None:
            blockSel["field"] = {"$in": fieldnames}
        if excludeFields is not None:
            blockSel["field"] = {"$nin": excludeFields}
        blockDocs = self.blockDB.find_blocks(blockSel)
        solver = SimplexScalarOffsetSolver(dbname=solverDBName,
                cname=solverCName,
                url=self.mosaicDB.url, port=self.mosaicDB.port)
        offsets = solver.find_best_offsets()
        print "Using offsets", offsets
        
        blockPath, weightPath = blockmosaic.block_mosaic(blockDocs, offsets,
                mosaicName, self.workDir,
                offset_fcn=offsettools.apply_offset)

        self.mosaicDB.collection.update({"_id": mosaicName},
                {"$set": {"image_path": blockPath,
                          "weight_path": weightPath}})

    def add_short_stacks(self, mosaicName, stacks):
        """Paste a stack directly onto the mosaic (made with `make_mosaic`.
        
        Stacks is a list of (stack sel, (ra, dec)) tuples, where (ra, dec)
        is the centre of the saturated region.
        """
        stackDB = StackDB()
        stackPaths = []
        for stackSel in stacks:
            stackDoc = stackDB.collection.find_one(stackSel)
            if stackDoc is None:
                print "Couldn't find a stack!"
                print stackSel
                continue
            stackPath = stackDoc['image_path']
            stackWeightPath = stackDoc['weight_path']
            stackPaths.append((stackPath, stackWeightPath))
        print stackPaths
        mosaicDoc = self.mosaicDB.collection.find_one({"_id": mosaicName})
        mosaicPath = mosaicDoc['image_path']
        mosaicWeightPath = mosaicDoc['weight_path']
        diffImageDir = os.path.join(self.workDir, "short_stack_diffs")
        paste_short_stack(mosaicPath, mosaicWeightPath, stackPaths,
                diffImageDir)

    def subsample_mosaic(self, mosaicName, pixelScale=1., fluxscale=True):
        """Subsamples the existing mosaic to 1 arcsec/pixel."""
        mosaicDoc = self.mosaicDB.collection.find_one({"_id": mosaicName})
        print "Mosaic Name:", mosaicName
        print "Mosaic Doc:", mosaicDoc
        fullMosaicPath = mosaicDoc['image_path']
        downsampledPath = blockmosaic.subsample_mosaic(fullMosaicPath,
                pixelScale=pixelScale, fluxscale=fluxscale)
        downsampledWeightPath = os.path.splitext(downsampledPath)[0] \
                + ".weight.fits"
        self.mosaicDB.collection.update({"_id": mosaicName},
                {"$set": {"subsampled_path": downsampledPath,
                          "subsampled_weight": downsampledWeightPath}})
        tiffPath = os.path.join(self.workDir, mosaicName + ".tif")
        subprocess.call("stiff -VERBOSE_TYPE QUIET %s -OUTFILE_NAME %s"
                % (downsampledPath, tiffPath), shell=True)


def paste_short_stack(mosaicPath, mosaicWeightPath, stacks, workDir):
    """Paste unsaturated pixels from stack into mosaic, adjusting for sky
    offset difference.
    """
    from difftools import ResampledWCS, Overlap, _computeDiff, SliceableImage
    mFITS = astropy.io.fits.open(mosaicPath)
    mwFITS = astropy.io.fits.open(mosaicWeightPath)
    mosaicFrame = ResampledWCS(mFITS[0].header)
    print "Mosaic stats", mFITS[0].data.min(), mFITS[0].data.max()
    for (stackPath, stackWeightPath) in stacks:
        sFITS = astropy.io.fits.open(stackPath)
        swFITS = astropy.io.fits.open(stackWeightPath)
        stackFrame = ResampledWCS(sFITS[0].header)
        overlap = Overlap(mosaicFrame, stackFrame)
        # Compute difference between stack and mosaic
        arg = ("mosaic", mosaicPath, mosaicWeightPath,
                "stack", stackPath, stackWeightPath,
                overlap, workDir, workDir)
        field1, field2, offsetData = _computeDiff(arg)
        meanDiff = offsetData['mean']
        # Offset mosaic to right level
        # meanDiff = mosaic - stack
        sImage = sFITS[0].data + meanDiff
        sWeight = swFITS[0].data
        # Make slices in overlap
        mosaicSL = SliceableImage('mosaic', mFITS[0].data, mwFITS[0].data)
        stackSL = SliceableImage('stack', sImage, sWeight)
        repPix = np.where((mosaicSL.weight <= 0.)
                & (stackSL.weight >= 0.))
        # Now apply the replacement to orginal mosaic image
        r = mosaicSL.r  # slice array
        mFITS[0].data[r[1][0]:r[1][1], r[0][0]:r[0][1]][repPix] = \
                stackSL.image[repPix]
        mwFITS[0].data[r[1][0]:r[1][1], r[0][0]:r[0][1]][repPix] = 1.
        sFITS.close()
        swFITS.close()
    mFITS.writeto(mosaicPath, clobber=True)
    mwFITS.writeto(mosaicWeightPath, clobber=True)
    mFITS.close()
    mwFITS.close()
