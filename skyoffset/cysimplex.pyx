#__docformat__ = "restructuredtext en"
# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************

# A collection of optimization algorithms.  Version 0.5
# CHANGES
#  Added fminbound (July 2001)
#  Added brute (Aug. 2002)
#  Finished line search satisfying strong Wolfe conditions (Mar. 2004)
#  Updated strong Wolfe conditions line search to use cubic-interpolation (Mar. 2004)
#
# Mods by JSick for M31 WIRCam project.
# Converted to Cython

import numpy
cimport numpy # cython

# data type of numpy simplex vectors
DTYPE_F = numpy.float

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef numpy.float_t DTYPE_F_t
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.

def nm_simplex(object objf, numpy.ndarray[DTYPE_F_t, ndim=2] simstart, float xtol=1e-4,
        float ftol=1e-4, maxiter=None, maxfun=None,
        full_output=0, disp=1, retall=0, callback=None):
    """Minimize a function using the downhill simplex algorithm.
    
    :Parameters:
    
      objf : instance of objective function class.
          Responds to objf.compute(simplex) to evaluate given a simplex
      simstart : ndarray, float type
          Initial guess, shaped N+1 by N
      callback : callable
          Called after each iteration, as callback(xk), where xk is the
          current parameter vector.
    
    :Returns: (xopt, {fopt, iter, funcalls, warnflag})
    
      xopt : ndarray
          Parameter that minimizes function.
      fopt : float
          Value of function at minimum: ``fopt = func(xopt)``.
      iter : int
          Number of iterations performed.
      funcalls : int
          Number of function calls made.
      warnflag : int
          1 : Maximum number of function evaluations made.
          2 : Maximum number of iterations reached.
      allvecs : list
          Solution at each iteration.
    
    *Other Parameters*:
    
      xtol : float
          Relative error in xopt acceptable for convergence.
      ftol : number
          Relative error in func(xopt) acceptable for convergence.
      maxiter : int
          Maximum number of iterations to perform.
      maxfun : number
          Maximum number of function evaluations to make.
      full_output : bool
          Set to True if fval and warnflag outputs are desired.
      disp : bool
          Set to True to print convergence messages.
      retall : bool
          Set to True to return list of solutions at each iteration.
    
    :Notes:
    
        Uses a Nelder-Mead simplex algorithm to find the minimum of
        function of one or more variables.
    
    """
    cdef unsigned int N = simstart.shape[1] # simplex is N+1 by N
    cdef unsigned int last = N - 1 # rather than using sim[-1] use sim[last]
    cdef float rho = 1.
    cdef float chi = 2.
    cdef float psi = 0.5
    cdef float sigma = 0.5
    one2np1 = range(1,N+1)
    
    # Eval objf for each vertex
    cdef numpy.ndarray[DTYPE_F_t, ndim=1] fsim = numpy.zeros((N+1,), numpy.float)
    cdef unsigned int k
    cdef unsigned int j
    for k in xrange(N+1):
        fsim[k] = objf.compute(simstart[k,:])
    
    cdef numpy.ndarray ind = numpy.argsort(fsim)
    fsim = numpy.take(fsim,ind,0)
    # sort so sim[0,:] has the lowest function value
    cdef numpy.ndarray[DTYPE_F_t, ndim=2] sim = numpy.take(simstart,ind,0)
    # TODO is it ok to use None in cython?
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200
    
    cdef int iterations = 1

    #cdef float xbar
    cdef numpy.ndarray[DTYPE_F_t, ndim=1] xbar # len of one vertex
    cdef int doshrink
    cdef numpy.ndarray[DTYPE_F_t, ndim=1] xr # len of one vertex
    cdef float fxr # objf at xr
    cdef numpy.ndarray[DTYPE_F_t, ndim=1] xe # len of one vertex
    cdef float fxe # objf at xe
    cdef numpy.ndarray[DTYPE_F_t, ndim=1] xc # len of one vertex
    cdef float fxc # objf at xc
    cdef numpy.ndarray[DTYPE_F_t, ndim=1] xcc # len of one vertex
    cdef float fxcc # objf at xcc
    
    while (objf.ncalls < maxfun and iterations < maxiter):
        if (numpy.max(numpy.ravel(numpy.abs(sim[1:]-sim[0]))) <= xtol \
            and numpy.max(numpy.abs(fsim[0]-fsim[1:])) <= ftol):
            break
        xbar = numpy.add.reduce(sim[:-1],0) / N
        xr = (1.+rho)*xbar - rho*sim[last]
        fxr = objf.compute(xr)
        doshrink = 0
        
        if fxr < fsim[0]:
            xe = (1+rho*chi)*xbar - rho*chi*sim[-1]
            fxe = objf.compute(xe)
            
            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else: # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else: # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1+psi*rho)*xbar - psi*rho*sim[-1]
                    fxc = objf.compute(xc)
                    
                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink=1
                else:
                    # Perform an inside contraction
                    xcc = (1-psi)*xbar + psi*sim[-1]
                    fxcc = objf.compute(xcc)
                    
                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1
                        
                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma*(sim[j] - sim[0])
                        fsim[j] = objf.compute(sim[j])
                        
        ind = numpy.argsort(fsim)
        sim = numpy.take(sim,ind,0)
        fsim = numpy.take(fsim,ind,0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        #if retall:
        #    allvecs.append(sim[0])
    
    x = sim[0]
    fval = numpy.min(fsim)
    warnflag = 0
    
    if objf.ncalls >= maxfun:
        warnflag = 1
        if disp:
            pass
            # print "Warning: Maximum number of function evaluations has "\
            #       "been exceeded."
    elif iterations >= maxiter:
        warnflag = 2
        if disp:
            pass
            # print "Warning: Maximum number of iterations has been exceeded"
    else:
        if disp:
            pass
            # print "Optimization terminated successfully."
            # print "         Current function value: %f" % fval
            # print "         Iterations: %d" % iterations
            # print "         Function evaluations: %d" % fcalls[0]
    
    
    # if full_output:
    #     retlist = x, fval, iterations, fcalls[0], warnflag
    #     if retall:
    #         retlist += (allvecs,)
    # else:
    #     retlist = x
    #     if retall:
    #         retlist = (x, allvecs)
    
    # TODO return fcalls
    #print "cysimplex fval:", fval
    retlist = x, fval, iterations, objf.ncalls, warnflag
    
    return retlist


