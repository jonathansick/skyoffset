import os
import logging
import platform
import time
import multiprocessing
import numpy
import pymongo

# Pure python/numpy
import simplex
from scalarobj import ScalarObjective

# Cython/numpy
import cyscalarobj
import cysimplex


class MultiStartSimplex(object):
    """Baseclass for multi-start recongerging simplex solvers."""
    def __init__(self, dbname, cname, url, port):
        #super(MultiStartSimplex, self).__init__()
        self.dbname, cname, url, port = dbname, cname, url, port
        self.dbname = dbname
        self.cname = cname
        self.url = url
        self.port = port

        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]

    def resetdb(self):
        """Delete existing entries in the mongodb collection for this
        multi simplex optimization."""
        # Drop the collection, then recreate it
        self.db.drop_collection(self.cname)
        self.collection = self.db[self.cname]

    def _prep_log_file(self):
        self.startTime = time.clock()  # for timing with close_log_file()
        logDir = os.path.dirname(self.logPath)
        if os.path.exists(logDir) is False: os.makedirs(logDir)
        logging.basicConfig(filename=self.logPath, level=logging.INFO)
        logging.info("STARTING NEW SIMPLEX OPTIMIZATION ====================")
        hostname = platform.node()
        now = time.localtime(time.time())
        timeStamp = time.strftime("%y/%m/%d %H:%M:%S %Z", now)
        logging.info("MultiStartSimplex started on %s at %s"
                % (hostname, timeStamp))
    
    def _close_log_file(self):
        endTime = time.clock()
        duration = (endTime - self.startTime) / 3600.
        logging.info("ENDING SIMPLEX OPTIMIZATION. Duration: %.2f hours"
                % duration)


class SimplexScalarOffsetSolver(MultiStartSimplex):
    """Uses a Multi-Start and Reconverging algorithm for converging on the
    the set of scalar sky offsets that minimize coupled image differences.

    The optimization is persisted in real-time to MongoDB. This means
    that multiple computers could be running threads and adding results
    to the same pool. While optimization is running, it is possible to
    query for the best-to-date offset solution.
    """
    def __init__(self, dbname="m31", cname="simplexscalar",
            url="localhost", port=27017):
        super(SimplexScalarOffsetSolver, self).__init__(dbname,
                cname, url, port)
    
    def multi_start(self, couplings, nTrials, logPath, initSigma=6e-10,
            restartSigma=1e-11, mp=True, cython=True):
        """Start processing using the Multi-Start Reconverging algorithm.
        :param nTrials: number of times a simplex is started.
        :param initSigma: dispersion of offsets
        :param restartSigma: dispersion of offsets about a converged point
            when making a restart simplex.
        :param mp: if True, run simplexes in parallel with `multiprocessing`.
        :param cython: True to use the cython version of simplex.
        """
        self.logPath = logPath
        self._prep_log_file()
        self.couplings = couplings
        if cython:
            self.objf = cyscalarobj.ScalarObjective(self.couplings)
        else:
            self.objf = ScalarObjective(self.couplings) # object function instance

        ndim = self.objf.get_ndim()
        xtol = 10.**(-4.) # fractional error in offsets acceptable for convergence
        ftol = 10.**(-4.) # fractional error in objective function acceptable for convergence
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
            sim = numpy.zeros([ndim+1, ndim], dtype=numpy.float64)
            for i in xrange(ndim+1):
                sim[i,:] = initSigma*numpy.random.standard_normal(ndim)
            args = [sim, cython, self.couplings, simplexArgs, restartSigma,
                xtol, n, nTrials, self.logPath, dbArgs]
            argsQueue.append(args)

        # Run the queue
        if mp:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            pool.map(_simplexWorker, argsQueue)
        else:
            map(_simplexWorker, argsQueue)

        self._close_log_file()
        
    def find_best_offsets(self):
        """Queries the mongodb collection of simplex runs to find the
        optimal result. Returns a dictionary of scalar offsets, keyed
        by the field name.
        """
        bestEnergy = 1e99 # running tally of best optimization result
        bestOffsets = {}
        recs = self.collection.find({}, ['best_fopt','best_offsets'])
        for rec in recs:
            if rec['best_fopt'] < bestEnergy:
                bestEnergy = rec['best_fopt']
                bestOffsets = rec['best_offsets']
        # Normalize these offsets so that the net offset is zero
        netOffset = 0.
        fieldCount = 0
        for field, offset in bestOffsets.iteritems():
            netOffset += offset
            fieldCount += 1
        print "Net offset %.2e" % netOffset
        netOffset = netOffset / fieldCount
        for field, offset in bestOffsets.iteritems():
            bestOffsets[field] = offset - netOffset
        return bestOffsets

def _simplexWorker(argsList):
    """multiprocessing worker function for doing multi-trial simplex solving.
    This essentially replaces the multi_start_simplex function in simplex.py
    
    But this exists because it implicitly specifies the target function for the
    optimization; multiprocessing can't pickle a function object.
    
    This simplex worker has the ability to restart at the site of convergence
    by constructing a simplex that is randomly distributed about the best vertex.
    The simplex keeps reconverging from perturbed simplex until the reconverged
    minimum matches the previous minimum. That is, I believe I have a global
    minimum if the simplex returns to where it started.
    """
    startTime = time.clock()
    sim, useCython, couplings, kwargs, restartSigma, xTol, n, nTrials, logFilePath, dbArgs = argsList
    if useCython:
            objf = cyscalarobj.ScalarObjective(couplings)
    else:
            objf = ScalarObjective(couplings) 
    # Choose the simplex code
    if useCython:
        nm_simplex = cysimplex.nm_simplex
    else:
        nm_simplex = simplex.nm_simplex
    #print "Running simplex %i/%i"% (n,nTrials)
    Ndim = sim.shape[1]
    _evalObjFunc = lambda offsets, objF: objF.compute(offsets)
    # These variables keep track of how the code performs
    totalFCalls = 0
    nRestarts = 0
    # Initial simplex compute
    _xOpt, _fOpt, _nIters, _nFcalls, _warnflag = nm_simplex(objf,
        sim, **kwargs)

    bestFOpt = _fOpt
    bestXOpt = _xOpt.copy()
    totalFCalls += _nFcalls
    # These arrays list the running tally of restarts vs best fopt vs total f calls
    restartTally = [nRestarts]
    bestFOptTally = [bestFOpt]
    totalFCallTally = [totalFCalls]
    # initiate restarts
    while True:
        nRestarts += 1
        sim = numpy.zeros([Ndim+1, Ndim], dtype=numpy.float64)
        sim[0,:] = bestXOpt.copy() # first vertex is the best point
        for i in xrange(1,Ndim+1): # rest are randomly distributed.
            sim[i,:] = restartSigma*numpy.random.standard_normal(Ndim) + bestXOpt
        _xOpt, _fOpt, _nIters, _nFcalls, _warnflag = nm_simplex(objf,
            sim, **kwargs)
        totalFCalls += _nFcalls
        # Ensure that the point has converged
        convergenceFrac = (_xOpt - bestXOpt) / bestXOpt
        if len(numpy.where(convergenceFrac > xTol)[0]) > 0:
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
    # Report this in the log
    runtime = time.clock() - startTime
    if logFilePath is not None:
        logging.basicConfig(filename=logFilePath,level=logging.INFO)
        logging.info("%i/%i converged to %.4e in %.2f minutes, %i local restarts" % (n, nTrials, bestFOpt, runtime/60., nRestarts))
    # Dictionary stores the history of restarts, as well as teh best solution
    # as a field offset dictionary (we're breaking reusability here... just
    # to make things faster.)
    convergenceHistory = {"total_calls": totalFCalls, "n_restarts": nRestarts,
        "runtime": runtime,
        "best_offsets": objf.get_best_offsets(),
        "best_fopt": bestFOpt,
        "restart_hist": restartTally,
        "fopt_hist": bestFOptTally,
        "fcall_hist": totalFCallTally}
    # Connect to MongoDB and add our convergence history!
    try:
        connection = pymongo.Connection(dbArgs['url'], dbArgs['port'])
        db = connection[dbArgs['dbname']]
        collection = db[dbArgs['cname']]
        collection.insert(convergenceHistory)
    except pymongo.errors.AutoReconnect:
        logging.info("pymongo.errors.AutoReconnect on %i"%n)
