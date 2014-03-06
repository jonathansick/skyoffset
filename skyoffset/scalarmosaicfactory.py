#!/usr/bin/env python
# encoding: utf-8
"""
Make mosaics using scalar sky offsets.
"""

import os
import numpy as np
import subprocess

from difftools import Couplings  # Scalar couplings
from multisimplex import SimplexScalarOffsetSolver
import blockmosaic  # common mosaic construction functions
import offsettools
from noisefactory import NoiseMapFactory


class ScalarMosaicFactory(object):
    """Pipeline class for solving scalar sky offsets between overlapping images
    and producing a mosaic.

    Parameters
    ----------
    mosaic_name : str
        Name of the mosaic being generated.
    block_sel : dict
        A MongoDB selector for blocks in the ``BlockDB``.
    blockdb : :class:`skyoffset.imagedb.MosaicDB` instance
        A MosaicDB database containing *blocks*, the component images that
        will be mosaiced together.
    mosaicdb : :class:`skyoffset.imagedb.MosaicDB` instance
        A MosaicDB database where the final mosaic will be stored.
    workdir : str
        Directory where the mosaic will be created. This directory will
        be created if necessary.
    swarp_configs : dict
        A dictionary of configurations to pass to
        :class:`moastro.astromatic.Swarp`.
    """
    def __init__(self, mosaic_name, block_sel, blockdb, mosaicdb,
            workdir, swarp_configs=None):
        super(ScalarMosaicFactory, self).__init__()
        self.mosaic_name = mosaic_name
        self.block_sel = dict(block_sel)
        self.blockdb = blockdb
        self.mosaicdb = mosaicdb
        self.workdir = os.path.join(workdir, mosaic_name)
        if not os.path.exists(workdir): os.makedirs(workdir)
        if swarp_configs:
            self._swarp_configs = dict(swarp_configs)
        else:
            self._swarp_configs = {}
    
    def solve_offsets(self, solver_dbname, solver_cname,
            n_runs=1000, dbmeta=None,
            reset_couplings=False, fresh_start=True,
            init_scale=5., restart_scale=2.):
        """Pipeline for solving the scalar sky offsets between a set of
        blocks.
        
        Parameters
        ----------
        solver_dbname : str
            Name of the MongoDB database where results from sky offset
            optimization are persisted.
        solver_cname : str
            Name of the MongoDB collection where results from sky offset
            optimization are persisted.
        n_runs : int
            Number of optimizations to start; the sky offsets from the best
            optimization run are chosen.
        init_scale : float
            Sets dispersion of initial guesses sky offsets as a fraction of
            the block difference dispersion.
        restart_scale : float
            Sets dispersion of sky offset simplex re-inflation as a fraction of
            the block difference dispersion after an optimization as converged.
        dbmeta : dict
            Arbitrary metadata to store in the mosaic's MongoDB document.
        reset_couplings : bool
            If ``True``, then the couplings (differences) between blocks
            will be recomputed.
        fresh_start : bool
            If ``True``, then previous optimization runs for this mosaic
            will be deleted from the sky offset solver MongoDB collection.
        """
        # Try to load a document with a previous run of this mosaic
        mosaic_doc = self.mosaicdb.find({"_id": self.mosaic_name}, one=True)
        if mosaic_doc is None:
            mosaic_doc = {"_id": self.mosaic_name}
            mosaic_doc.update(dbmeta)  # add meta data for this mosaic
            self.mosaicdb.c.insert(mosaic_doc)
        
        block_docs = self.blockdb.find_dict(self.block_sel)
        
        # Make couplings
        if 'couplings' not in mosaic_doc or reset_couplings:
            couplings = self._make_couplings(block_docs)
        else:
            couplings = self._reload_couplings(mosaic_doc['couplings'])
        
        # Precompute the output WCS; add to FootprintDB
        self._precompute_mosaic_footprint(block_docs, self.workdir)

        # Retrieve the ResampledWCS for blocks and mosaic
        mosaic_wcs = self.mosaicdb.make_resampled_wcs(
            {"_id": self.mosaic_name})
        block_wcss = {}
        for block_name, block_doc in block_docs.iteritems():
            sel = {"_id": block_name}
            block_wcss[block_name] = self.blockdb.make_resampled_wcs(sel)

        self.mosaicdb.c.update({"_id": self.mosaic_name},
                {"$set": {"solver_cname": self.mosaic_name,
                          "solver_dbname": solver_dbname}})

        # Perform optimization
        self._solve_offsets(self.mosaic_name, solver_dbname, couplings,
                block_wcss, mosaic_wcs, init_scale, restart_scale,
                fresh_start=fresh_start, n_runs=n_runs)

    def _reload_couplings(self, couplings_doc):
        """Attempt to create a CoupledPlanes instance from a MongoDB
        persisted document."""
        return Couplings.load_doc(couplings_doc)
    
    def _make_couplings(self, block_docs):
        """Computes the couplings between block_docs.
        :return: a difftools.Couplings instance.
        """
        couplings = Couplings()
        for block_name, block_doc in block_docs.iteritems():
            blockPath = block_doc['image_path']
            blockWeightPath = block_doc['weight_path']
            couplings.add_field(block_name, blockPath, blockWeightPath)
        diffImageDir = os.path.join(self.workdir, "diffs")
        couplings.make(diffImageDir)
        couplings_doc = couplings.get_doc()
        self.mosaicdb.c.update({"_id": self.mosaic_name},
                {"$set": {"couplings": couplings_doc}})
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
            fresh_start=True, n_runs=1000):
        """Use SimplexScalarOffsetSolver to derive offsets for this block."""
        logPath = os.path.join(self.workdir, "%s.log" % mosaicName)
        solver = SimplexScalarOffsetSolver(dbname=solverDBName,
                cname=mosaicName,
                url=self.mosaicdb.url, port=self.mosaicdb.port)

        if fresh_start:
            solver.resetdb()
        initSigma, resetSigma = self._simplex_dispersion(initScale,
                restartScale, couplings)
        solver.multi_start(couplings, n_runs, logPath, cython=True, mp=True,
                initSigma=initSigma,
                restartSigma=resetSigma)

        offsets = solver.find_best_offsets()

        # Estimate uncertainty in the zeropoint of the sky offsets
        zp_sigma = self._compute_offset_zp_sigma(offsets)

        self.mosaicdb.c.update({"_id": mosaicName},
                {"$set": {"offsets": offsets,
                    "offset_zp_sigma": zp_sigma,
                    "solver_cname": mosaicName,
                    "solver_dbname": solverDBName}})
        return solver
    
    def _precompute_mosaic_footprint(self, blockDocs, workDir):
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
        self.mosaicdb.add_footprint_from_header(self.mosaic_name, header)

    def _compute_offset_zp_sigma(self, offsets):
        """The offsets have a net zeropoint uncertainty due to the assumption
        that the net offset should be zero (i.e. error of the mean).
        """
        delta = np.array([offsets[k] for k in offsets])
        n_blocks = len(offsets)
        sigma = delta.std() / np.sqrt(float(n_blocks))
        return float(sigma)

    def make_mosaic(self, block_selector=None, target_fits=None):
        """Swarp a mosaic using the optimal sky offsets.
        
        The mosaic can be made anytime once entries are added
        to the solver's collection. This is because we initialize
        a SimplexScalarOffsetSolver that re-generates the list of best
        offsets from the collection of solver documents.

        Parameters
        ----------
        block_selector : dict
            An alternative MongoDB block selector (used instead of the one
            specific during instance initialization). This can be useful
            for building a mosaic with a subset of the blocks.
        target_fits : str
            Set to the path of a FITS file that will be used to define the
            output frame of the block. The output blocks will then correspond
            pixel-to-pixel. Note that both blocks should already be resampled
            into the same pixel space.
        """
        mosaicDoc = self.mosaicdb.c.find_one({"_id": self.mosaic_name})
        solver_cname = mosaicDoc['solver_cname']
        solver_dbname = mosaicDoc['solver_dbname']

        if block_selector:
            block_sel = dict(block_selector)
        else:
            block_sel = dict(self.block_sel)
        bloc_docs = self.blockdb.find_dict(block_sel)
        solver = SimplexScalarOffsetSolver(dbname=solver_dbname,
                cname=solver_cname,
                url=self.mosaicdb.url, port=self.mosaicdb.port)
        offsets = solver.find_best_offsets()
        
        mosaic_path, mosaic_weight_path = blockmosaic.block_mosaic(bloc_docs,
                offsets, self.mosaic_name, self._swarp_configs, self.workdir,
                target_fits=target_fits,
                offset_fcn=offsettools.apply_offset)

        self.mosaicdb.c.update({"_id": self.mosaic_name},
                {"$set": {"image_path": mosaic_path,
                          "weight_path": mosaic_weight_path}})

    def make_noisemap(self, block_selector=None):
        """Make a Gaussian sigma noise map, propagating those from stacks.

        Parameters
        ----------
        block_selector : dict
            An alternative MongoDB block selector (used instead of the one
            specific during instance initialization). This can be useful
            for building a mosaic with a subset of the blocks.
        """
        noise_paths, weight_paths = [], []
        if block_selector:
            block_sel = dict(block_selector)
        else:
            block_sel = dict(self.block_sel)
        block_docs = self.blockdb.find_dict(block_sel)
        for blockName, blockDoc in block_docs.iteritems():
            noise_paths.append(blockDoc['noise_path'])
            weight_paths.append(blockDoc['weight_path'])
        mosaic_doc = self.mosaicdb.find({"_id": self.mosaic_name}, one=True)
        mosaic_path = mosaic_doc['image_path']
        factory = NoiseMapFactory(noise_paths, weight_paths, mosaic_path,
                swarp_configs=dict(self._swarp_configs),
                delete_temps=True)
        self.mosaicdb.set(self.mosaic_name, "noise_path", factory.map_path)

    def subsample_mosaic(self, pixel_scale=1., fluxscale=True):
        """Subsamples the existing mosaic."""
        mosaicDoc = self.mosaicdb.c.find_one({"_id": self.mosaic_name})
        print "Mosaic Name:", self.mosaic_name
        print "Mosaic Doc:", mosaicDoc
        fullMosaicPath = mosaicDoc['image_path']
        downsampledPath = blockmosaic.subsample_mosaic(fullMosaicPath,
                self._swarp_configs,
                pixel_scale=pixel_scale, fluxscale=fluxscale)
        downsampledWeightPath = os.path.splitext(downsampledPath)[0] \
                + ".weight.fits"
        self.mosaicdb.c.update({"_id": self.mosaic_name},
                {"$set": {"subsampled_path": downsampledPath,
                          "subsampled_weight": downsampledWeightPath}})
        tiffPath = os.path.join(self.workdir, self.mosaic_name + ".tif")
        subprocess.call("stiff -VERBOSE_TYPE QUIET %s -OUTFILE_NAME %s"
                % (downsampledPath, tiffPath), shell=True)

    def make_tiff(self):
        """Render a tiff image of this block."""
        mosaicDoc = self.mosaicdb.c.find_one({"_id": self.mosaic_name})
        downsampledPath = mosaicDoc['image_path']
        tiffPath = os.path.join(self.workdir, self.mosaic_name + ".tif")
        subprocess.call("stiff -VERBOSE_TYPE QUIET %s -OUTFILE_NAME %s"
                % (downsampledPath, tiffPath), shell=True)
