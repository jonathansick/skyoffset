#!/usr/bin/env python
# encoding: utf-8
"""
Multi-start simplex controller for Planar Sky Offsets

History
-------
2011-07-28 - Created by Jonathan Sick

"""

__all__ = ['']

import os
import logging
import platform
import time
import multiprocessing
import numpy
import pymongo

# Cython/numpy
import cyplanarobj
import cysimplex

from multisimplex import MultiStartSimplex
import difftools

class PlanarMultiStartSimplex(MultiStartSimplex):
    """Multi-start reconvering simplex runner for planar offsets."""
    def __init__(self, dbname="m31", cname="simplexplanar",
            url="localhost", port=27017):
        super(PlanarMultiStartSimplex, self).__init__(dbname,
                cname, url, port)
    
    def multi_start(self, couplings, nTrials, logPath, initSigma=5e-9,
            slopeSigma=5e-13, restartSigma=1e-10, mp=True, cython=True):
        """Runs a multi-start, converging, planar simplex optimization.
        
        .. todo:: ensure that slopeSigma corresponds to the distribution
            of slope offsets.
        """
        self.logPath = logPath
        self._prep_log_file()
        self.couplings = couplings
        self.objf = cyplanarobj.PlanarObjective(self.couplings,
                levelScale=1., slopeScale=initSigma/slopeSigma)

        ndim = self.objf.get_ndim()
        xtol = 10.**(-20.) # fractional error in offsets acceptable for convergence
        ftol = 10.**(-20.) # fractional error in objective function acceptable for convergence
        maxiter = 1000 * ndim
        maxEvals = 1000 * ndim
        simplexArgs = {'xtol':xtol, 'ftol':ftol, 'maxiter':maxiter,
            'maxfun':maxEvals, 'full_output':True, 'disp':True,
            'retall':False, 'callback':None}
        dbArgs = {'dbname':self.dbname, 'cname':self.cname, 'url':self.url,
                'port':self.port}

        # Create initial simplexes
        argsQueue = []
        for n in xrange(nTrials):
            sim = self.objf.randomize_simplex(initSigma)
            args = (sim, cython, self.couplings, simplexArgs, restartSigma,
                    initSigma, slopeSigma, n, nTrials,
                    self.logPath, dbArgs)
            argsQueue.append(args)
        
        # sim, useCython, couplings, simplexArgs, restartSigma, initSigma, \
        #    slopeSigma, runID, nTrials, logFilePath, dbArgs = args

        #mp = False # DEBUG
        if mp:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            pool.map(_simplex_worker, argsQueue)
        else:
            map(_simplex_worker, argsQueue)

        self._close_log_file()

    def find_best_offsets(self, blockWCSs, mosaicWCS):
        """Get the best offset dictionary from the starts currently available
        in the solver collection.
        
        Offsets are packed as a field dictionary of (mx, my, c) planes.

        The offsets are normalized so that no net flux, nor net tilt
        is added to the mosaic.

        :param blockWCSs: dictionary of field: ResampledWCS for blocks
        :param mosaicWCS: the ResampledWCS computed for the mosaic
        """
        bestEnergy = 1e99 # running tally of bst optimization result
        bestOffsets = {}
        recs = self.collection.find({}, ['best_fopt','best_offsets'])
        # Get offsets with lowest energy from the solver DB
        for rec in recs:
            if rec['best_fopt'] < bestEnergy:
                bestEnergy = rec['best_fopt']
                bestOffsets = rec['best_offsets']
        # Normalize these offsets for no net offset or tilt
        normalizedOffsets = self._normalize_offsets(bestOffsets,
                blockWCSs, mosaicWCS)
        return normalizedOffsets
        #return bestOffsets

    def _normalize_offsets(self, offsets, blockWCSs, mosaicWCS):
        """Normalize the net tilt and offset of a planar offset dictionary."""
        netPlane = self._net_flux_plane(offsets)
        normalizedOffsets = {}
        print "blockWCSs"
        print blockWCSs
        for field, offsetPlane in offsets.iteritems():
            blockWCS = blockWCSs[field]
            normalizedOffsets[field] = self._subtract_plane(field,
                    offsetPlane, netPlane, blockWCS, mosaicWCS)
        return normalizedOffsets

    def _net_flux_plane(self, offsets):
        """Compute the net flux plane given a dictionary of offsets.
        
        The net flux plane is computed at the average of all slopes
        and the average of all offset levels
        """
        netMx = 0.
        netMy = 0.
        netC = 0.
        n = len(offsets.keys())
        for field, (mx, my, c) in offsets.iteritems():
            netMx += mx
            netMy += my
            netC += c
        netPlane = (netMx/n, netMy/n, netC/n)
        print "net_flux_plane", netPlane
        return netPlane

    def _subtract_plane(self, fieldname, plane, netPlane, blockWCS, mosaicWCS):
        """Computes plane - netPlane. Used for subtracting net offset planes.
        
        The 'wcs' objects are `ResampledWCS` instances, and needed to find
        the distance between the centre of the mosaic and the center of
        the mosaic.

        In this formalism, we treat the netPlane as belonging to the mosaic
        frame
        """
        overlap = difftools.Overlap(blockWCS, mosaicWCS)
        # Dx,Dy are distance from centre of netPlane(/mosaic) to centre of
        # the block
        Dx, Dy = overlap.getLowerCentreTrans()
        print "net plane sub for", fieldname
        print fieldname, Dx, Dy
        netPTrans = (netPlane[0], netPlane[1],
                netPlane[2] + netPlane[0]*Dx + netPlane[1]*Dy)
        # Subtract the transformed net plane
        normPlane = (plane[0] - netPTrans[0], plane[1] - netPTrans[1],
                plane[2] - netPTrans[2])
        return normPlane


def _simplex_worker(args):
    """Worker for PlanarMultiStartSimplex"""
    startTime = time.clock()
    
    sim, useCython, couplings, simplexArgs, restartSigma, initSigma, \
            slopeSigma, runID, nTrials, logFilePath, dbArgs = args
    
    runtime = time.clock() - startTime
    
    nm_simplex = cysimplex.nm_simplex
    objf = cyplanarobj.PlanarObjective(couplings,
                levelScale=1., slopeScale=initSigma/slopeSigma)

    # Initialize counters
    totalFCalls = 0
    nRestarts = 0

    # Initial simplex run
    _xOpt, _fOpt, _nIters, _nFcalls, _warnflag = nm_simplex(objf,
        sim, **simplexArgs)

    bestFOpt = _fOpt
    bestXOpt = _xOpt.copy()
    totalFCalls += _nFcalls
    # These arrays list the running tally of restarts vs best fopt vs total f calls
    restartTally = [nRestarts]
    bestFOptTally = [bestFOpt]
    totalFCallTally = [totalFCalls]

    # Repetitively restart until definitive convergence
    while True:
        nRestarts += 1
        objf.restart_simplex(bestXOpt, restartSigma)
        _xOpt, _fOpt, _nIters, _nFcalls, _warnflag = nm_simplex(objf,
        sim, **simplexArgs)
        totalFCalls += _nFcalls
        # Ensure that the point has converged
        convergenceFrac = (_xOpt - bestXOpt) / bestXOpt
        if len(numpy.where(convergenceFrac > simplexArgs['xtol'])[0]) > 0:
            # do another restart of the simplex
            if _fOpt < bestFOpt:
                # but we did find a new minimum
                bestFOpt = _fOpt
                bestXOpt = _xOpt.copy()
                restartTally.append(nRestarts)
                bestFOptTally.append(bestFOpt)
                totalFCallTally.append(totalFCalls)
        else:
            # we're converged
            break
    
    runtime = time.clock() - startTime
    if logFilePath is not None:
        logging.basicConfig(filename=logFilePath,level=logging.INFO)
        logging.info("%i/%i converged to %.4e in %.2f minutes, %i local restarts" % (runID, nTrials, bestFOpt, runtime/60., nRestarts))
    
    # Dictionary stores the history of restarts, as well as the
    # best solution
    convergenceHistory = {"total_calls": totalFCalls, "n_restarts": nRestarts,
        "runtime": runtime,
        "best_offsets": objf.get_best_offsets(),
        "best_fopt": bestFOpt,
        "restart_hist": restartTally,
        "fopt_hist": bestFOptTally,
        "fcall_hist": totalFCallTally}
    print "RAN %i calls, %i restarts //////////////" % (totalFCalls, nRestarts)
    print "//////////// best_offsets", objf.get_best_offsets()
    # Connect to MongoDB and add our convergence history!
    try:
        connection = pymongo.Connection(dbArgs['url'], dbArgs['port'])
        db = connection[dbArgs['dbname']]
        collection = db[dbArgs['cname']]
        collection.insert(convergenceHistory)
    except pymongo.errors.AutoReconnect:
        logging.info("pymongo.errors.AutoReconnect on %i"%runID)
