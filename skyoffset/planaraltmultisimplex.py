#!/usr/bin/env python
# encoding: utf-8
"""
Multi-start simplex controller for Planar Sky Offsets using the alternating
algorithm for converging on slopes and levels of offset planes.

History
-------
2011-08-17 - Created by Jonathan Sick

"""

__all__ = ['']
import logging
import time
import multiprocessing
import numpy
import pymongo

# Cython/numpy
import cyplanarobj
import cysimplex

from planarmultisimplex import PlanarMultiStartSimplex
import difftools

class AltPlanarMultiStartSimplex(PlanarMultiStartSimplex):
    """docstring for AltPlanarMultiSimplex"""
    def __init__(self, **kwargs):
        super(AltPlanarMultiStartSimplex, self).__init__(**kwargs)
        
    def multi_start(self, couplings, initLevels, nTrials, logPath, levelSigma=5e-9,
            slopeSigma=5e-13, restartFraction=0.5, mp=True, cython=True,
            nThreads=multiprocessing.cpu_count()):
        """Runs a multi-start, converging, planar simplex optimization.
        
        .. todo:: ensure that slopeSigma corresponds to the distribution
            of slope offsets.
        """
        self.logPath = logPath
        self._prep_log_file()
        self.couplings = couplings

        xtol = 1e-8 # fractional error in offsets
        ftol = 1e-8 # fractional error in objective function
        # expect that the worker will fill in the maxiter and maxfun
        simplexArgs = {'xtol':xtol, 'ftol':ftol,
                'full_output':True, 'disp':True,
                'retall':False, 'callback':None}
        dbArgs = {'dbname':self.dbname, 'cname':self.cname, 'url':self.url,
                'port':self.port}

        # Create simplex starts
        argsQueue = []
        #print "using initLevels:", initLevels
        for n in xrange(nTrials):
            initPlanes = initialize_plane(couplings, initLevels, levelSigma, slopeSigma)
            time.sleep(0.1) # to prevent the random number generator from correllating
            args = (self.couplings, initPlanes, simplexArgs, levelSigma,
                    slopeSigma, restartFraction, n, nTrials, self.logPath,
                    dbArgs)
            argsQueue.append(args)

        #mp = False
        if mp:
            pool = multiprocessing.Pool(processes=nThreads)
            pool.map(simplex_worker, argsQueue)
        else:
            map(simplex_worker, argsQueue)

        self._close_log_file()


def simplex_worker(args):
    """Carries out the alternating planar optimization."""
    maxAltIters = 10000
    startTime = time.clock()
    couplings, initPlane, simplexArgs, levelSigma, \
            slopeSigma, restartFraction, runID, nTrials, logPath, \
            dbArgs = args
    nFields = len(couplings.fields.keys())
    simplexArgs['maxiter'] = 1000 * nFields
    simplexArgs['maxfun'] = 1000 * nFields
    xtol = simplexArgs['xtol']
    totalFCalls = 0
    totalRestarts = 0
    levelFopt = 0.
    slopeFopt = 0.
    #initPlane = initialize_plane(couplings, initLevels, levelSigma, slopeSigma)
    newPlane, stats = run_opt(couplings, initPlane, levelSigma, slopeSigma,
            restartFraction, simplexArgs)
    lastPlane = initPlane
    altIters = 1
    totalFCalls += stats['n_fcalls']
    totalRestarts += stats['n_restarts']
    levelFopt = stats['level_fopt']
    slopeFopt = stats['slope_fopt']
    while is_converged(lastPlane, newPlane, xtol) is False:
        altIters += 1
        lastPlane = newPlane
        newPlane, stats = run_opt(couplings, lastPlane, levelSigma,
                slopeSigma, restartFraction, simplexArgs)
        totalFCalls += stats['n_fcalls']
        totalRestarts += stats['n_restarts']
        levelFopt = stats['level_fopt']
        slopeFopt = stats['slope_fopt']
        if altIters == maxAltIters: break
        
    runtime = time.clock() - startTime

    # Dictionary stores the history of restarts, as well as the
    # best solution
    convergenceHistory = {
        "total_calls": totalFCalls, "n_restarts": totalRestarts,
        "runtime": runtime,
        "best_offsets": newPlane,
        "best_fopt": levelFopt,
        "slope_fopt": slopeFopt,
        "alt_iters": altIters,
        "init_plane": initPlane}
    # Connect to MongoDB and add our convergence history!
    try:
        connection = pymongo.Connection(dbArgs['url'], dbArgs['port'])
        db = connection[dbArgs['dbname']]
        collection = db[dbArgs['cname']]
        collection.insert(convergenceHistory)
    except pymongo.errors.AutoReconnect:
        logging.info("pymongo.errors.AutoReconnect on %i"%runID)


def initialize_plane(couplings, initLevels, levelSigma, slopeSigma):
    """Create a random set of planar offsets to begin the optimization."""
    fields = couplings.fields.keys()
    planes = {}
    for i, field in enumerate(fields):
        mx = float(slopeSigma * numpy.random.randn())
        my = float(slopeSigma * numpy.random.randn())
        c = float(levelSigma * numpy.random.randn()+initLevels[field])
        #c = float(levelSigma * numpy.random.randn())
        planes[field] = (mx, my, c)
    return planes
    
def run_opt(couplings, lastPlane, levelSigma, slopeSigma, restartFrac,
        simplexArgs):
    """Runs sequential level then slope optimization."""
    totalFCalls, totalRestarts = 0, 0
    # Run level optimization
    levelObj = cyplanarobj.AltLevelObjective(couplings, lastPlane)
    sim = levelObj.randomize_simplex(levelSigma)
    #print "Made initial levelObj.randomize_simplex"
    revPlane, levelFOpt, _fCalls, _nRestarts = drive_restart_simplex(
        levelObj, sim, levelSigma*restartFrac, simplexArgs)
    totalFCalls += _fCalls
    totalRestarts += _nRestarts
    #print "ran level opt"
    # Run slope optimization
    slopeObj = cyplanarobj.AltSlopeObjective(couplings, revPlane)
    sim = slopeObj.randomize_simplex(slopeSigma)
    newPlane, slopeFOpt, _fCalls, _nRestarts = drive_restart_simplex(
            slopeObj, sim, slopeSigma*restartFrac, simplexArgs)
    totalFCalls += _fCalls
    totalRestarts += _nRestarts
    stats = {"n_fcalls": totalFCalls, "n_restarts": totalRestarts,
            "level_fopt": levelFOpt, "slope_fopt": slopeFOpt}
    return newPlane, stats

def is_converged(lastPlane, newPlane, xtol):
    """Checks that all planes have fractionally converged, within xtol."""
    # Convert planes to numpy arrays
    fields = lastPlane.keys()
    n = len(fields)
    lastPlaneArray = numpy.zeros([n,3], dtype=numpy.float)
    newPlaneArray = numpy.zeros([n,3], dtype=numpy.float)
    for i, field in enumerate(fields):
        lastPlaneArray[i,:] = lastPlane[field]
        newPlaneArray[i,:] = newPlane[field]
    conv = 1. - newPlaneArray / lastPlaneArray
    conv = numpy.sqrt(conv**2)
    if len(numpy.where(conv > xtol)[0]) > 0:
        #print "Planes not converged"
        return False
    else:
        #print "Planes converged"
        return True

def drive_restart_simplex(objf, sim, restartSigma, simplexArgs):
    """Drive the restarting simplex for both level and plane objectives."""
    totalFCalls = 0
    nRestarts = 0

    # Initial simplex run
    _xOpt, _fOpt, _nIters, _nFcalls, _warnflag = cysimplex.nm_simplex(objf,
        sim, **simplexArgs)
    #print "Ran initial simplex"

    bestFOpt = _fOpt
    bestXOpt = _xOpt.copy()
    totalFCalls += _nFcalls

    # Repetitively restart until definitive convergence
    while True:
        nRestarts += 1
        sim = objf.restart_simplex(bestXOpt, restartSigma)
        _xOpt, _fOpt, _nIters, _nFcalls, _warnflag = cysimplex.nm_simplex(objf,
            sim, **simplexArgs)
        #print "new fopt: %.2e" % _fOpt
        totalFCalls += _nFcalls
        # Ensure that the point has converged
        convergenceFrac = (_xOpt - bestXOpt) / bestXOpt
        if len(numpy.where(convergenceFrac > simplexArgs['xtol'])[0]) > 0:
            # do another restart of the simplex
            if _fOpt < bestFOpt:
                # but we did find a new minimum
                bestFOpt = _fOpt
                bestXOpt = _xOpt.copy()
        else:
            # we're converged
            break
    planes = objf.get_best_offsets()
    #print "Restarting simplex converged! fopt is:", bestFOpt
    return planes, bestFOpt, totalFCalls, nRestarts



