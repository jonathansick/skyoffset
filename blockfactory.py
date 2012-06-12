#!/usr/bin/env python
# encoding: utf-8
"""
Build blocks from stacks

2012-05-18 - Created by Jonathan Sick
"""


import os
import shutil
import multiprocessing
import subprocess
import numpy as np

from moastro.astromatic import Swarp

from difftools import Couplings
from multisimplex import SimplexScalarOffsetSolver
import offsettools


class BlockFactory(object):
    """Build froms from sets of stacks within a field."""
    def __init__(self, blockname, blockDB, stackDB, workDir):
        super(BlockFactory, self).__init__()
        self.blockDB = blockDB
        self.stackDB = stackDB
        self.workDir = workDir
        if not os.path.exists(workDir): os.makedirs(workDir)
        self.blockname = blockname

    def build(self, stackSelector, solverCName,
            freshStart=True, dbMeta={}, instrument="WIRCam",
            solverDBName="skyoffsets", mp=False):
        """
        :param fieldname: the field of this block
        :param band: the FILTER of stacks
        :param workDir: directory where blocks are made, the block for this
            field is actually made in workDir/fieldname_band
        :param solverCName: name of the MongoDB collection to store the
            real-time optimization results in from individual runs.
        :param freshStart: True, will cause the solver's mongodb collection
            to be reset. Set to False to build upon results of previous
            solver runs.
        """
        self.mp = mp  # Bool flag to use multiprocessing
        stackDocs = self.stackDB.find_stacks(stackSelector)
        # Make couplings
        couplings = self._make_couplings(self.blockname, stackDocs)
        # Add this block to the BlockDB collection
        doc = {"_id": self.blockname,
               "solver_cname": solverCName,
               "solver_dbname": solverDBName,
               "instrument": instrument}
        print "Inserting doc", doc
        doc.update(dbMeta)
        self.blockDB.insert(doc)
        # Perform optimization
        solver = self._solve_offsets(self.blockname, stackDocs, couplings,
                solverDBName, solverCName, freshStart=freshStart)
        offsets = solver.find_best_offsets()  # offset dictionary
        # Insert into BlockDB
        self.blockDB.update(self.blockname, "offsets", offsets)

    def _make_couplings(self, blockName, stackDocs):
        """Computes the couplings between stackDocs.
        :return: a difftools.Couplings instance.
        """
        couplings = Couplings()
        for stackName, stackDoc in stackDocs.iteritems():
            stackPath = stackDoc['image_path']
            stackWeightPath = stackDoc['weight_path']
            couplings.add_field(stackName, stackPath, stackWeightPath)
        diffImageDir = os.path.join(self.workDir, "diffs")
        couplings.make(diffImageDir)
        # pairNames, diffs, diffSigmas, allFields = couplings.get_field_diffs()
        shutil.rmtree(diffImageDir)
        return couplings
    
    def _solve_offsets(self, blockName, stackDocs, couplings,
            solverDBName, solverCName, freshStart=True):
        """Use SimplexScalarOffsetSolver to derive offsets for this block."""
        logPath = os.path.join(self.workDir, "%s.log" % blockName)
        solver = SimplexScalarOffsetSolver(dbname=solverDBName,
                cname=solverCName,
                url=self.blockDB.url, port=self.blockDB.port)
        if freshStart:
            solver.resetdb()
        initSigma, resetSigma = self._simplex_dispersion(couplings)
        solver.multi_start(couplings, 1000, logPath, mp=self.mp, cython=True,
                initSigma=initSigma,
                restartSigma=resetSigma)
        return solver

    def _simplex_dispersion(self, couplings):
        """Estimate the standard deviation (about zero offset) to initialize
        the simplex dispersion around.

        Return
        ------
        `initialSigma` and `restartSigma`.
        """
        diffList = [diff for k, diff in couplings.fieldDiffs.iteritems()]
        diffList = np.array(diffList)
        diffSigma = diffList.std()
        return 2 * diffSigma, diffSigma
    
    def make_block_mosaic(self, stackDB, stackSel, blockName, workDir):
        """The block mosaic can be made anytime once entries are added
        to the solver's collection."""
        self.workDir = os.path.join(workDir, blockName)
        blockDoc = self.blockDB.find_one({"_id": blockName})
        print blockDoc
        solverCName = blockDoc['solver_cname']
        solverDBName = blockDoc['solver_dbname']
        stackDocs = stackDB.find_stacks(stackSel)
        solver = SimplexScalarOffsetSolver(dbname=solverDBName,
                cname=solverCName, url=self.blockDB.url,
                port=self.blockDB.port)
        offsets = solver.find_best_offsets()

        self.offsetDir = os.path.join(self.workDir, "offset_images")
        if os.path.exists(self.offsetDir) is False: os.makedirs(self.offsetDir)

        # apply offsets to images, then swarp
        args = []
        offsetPaths = {}
        weightPaths = {}
        fieldNames = []
        for stackName, stackDoc in stackDocs.iteritems():
            print "stackDoc:", stackDoc
            stackName = stackDoc['_id']
            fieldNames.append(stackName)
            imagePath = stackDoc['image_path']
            weightPaths[stackName] = stackDoc['weight_path']
            offset = offsets[stackName]
            offsetPath = os.path.join(self.offsetDir, stackName + ".fits")
            offsetPaths[stackName] = offsetPath
            arg = (stackName, imagePath, offset, offsetPath)
            args.append(arg)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pool.map(offsettools.apply_offset, args)
        
        imagePathList = [offsetPaths[k] for k in fieldNames]
        weightPathList = [weightPaths[k] for k in fieldNames]

        # By not resampling, it means that the block will still be in the
        # same resampling system as all other blocks (hopefully)
        swarpConfigs = {"WEIGHT_TYPE": "MAP_WEIGHT",
                "COMBINE_TYPE": "WEIGHTED",
                "RESAMPLE": "N",
                "COMBINE": "Y"}
        swarp = Swarp(imagePathList, blockName,
                weightPaths=weightPathList, configs=swarpConfigs,
                workDir=self.workDir)
        swarp.run()
        blockPath, weightPath = swarp.mosaic_paths()
        self.blockDB.update(blockName, "image_path", blockPath)
        self.blockDB.update(blockName, "weight_path", weightPath)

        # Delete offset images
        shutil.rmtree(self.offsetDir)

    def make_tiff(self, workDir):
        """Render a tiff image of this block into workDir"""
        blockDoc = self.blockDB.find_one({"_id": self.blockname})
        downsampledPath = blockDoc['image_path']
        tiffPath = os.path.join(self.workDir, self.blockname + ".tif")
        subprocess.call("stiff -VERBOSE_TYPE QUIET %s -OUTFILE_NAME %s"
                % (downsampledPath, tiffPath), shell=True)
