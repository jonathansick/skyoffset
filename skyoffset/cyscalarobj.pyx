"""Scalar sky offset objective function (pure python version)"""
import numpy
cimport numpy

#from difftools import Couplings

DTYPE_F = numpy.float
DTYPE_I = numpy.int
# Each numpy array class member needs its own ctypedef for pointer assignment
ctypedef numpy.int_t DTYPE_I_t
ctypedef numpy.float_t DTYPE_F_t
ctypedef numpy.int_t DTYPE_UI_t
ctypedef numpy.int_t DTYPE_LI_t
ctypedef numpy.float_t DTYPE_D_t
ctypedef numpy.float_t DTYPE_DS_t
ctypedef numpy.float_t DTYPE_DA_t
ctypedef numpy.float_t DTYPE_W_t
ctypedef Py_ssize_t INDEX_t

cdef class ScalarObjective:
    """Objective function for scalar sky offsets.
    :param couplings: difftools couplings object
    :param initOffsets: optional dictionary of offsets, keyed by image key.
        Any offset optimized will be **relative** to these initial offsets.
        That is, these offsets are *corrections* to information in the couplings
        object.
    :param members: limit objective function to be sensitive to a limited
        set of images. Leave as None if all images in the `couplings` should
        be considered.
    :param scale: number that physical offsets are multiplied by, so that
        the optimization can operate on values closer to unity.
    """
    cdef list members
    cdef DTYPE_F_t scale
    cdef DTYPE_F_t bestEnergy
    cdef INDEX_t nTerms
    cdef public int ncalls
    # Pointers into numpy array members
    cdef DTYPE_UI_t* tUpIndex_p
    cdef DTYPE_LI_t* tLoIndex_p
    cdef DTYPE_D_t* tDelta_p
    cdef DTYPE_DS_t* tDeltaSigma_p
    cdef DTYPE_DA_t* tDeltaArea_p
    cdef DTYPE_W_t* tWeight_p
    # Declare numpy array members
    cdef numpy.ndarray tUpIndex
    cdef numpy.ndarray tLoIndex
    cdef numpy.ndarray tDelta
    cdef numpy.ndarray tDeltaSigma
    cdef numpy.ndarray tDeltaArea
    cdef numpy.ndarray tWeight

    cdef numpy.ndarray bestOffsets


    def __init__(self, object couplings, dict initOffsets={},
            list members=[], float scale=1.):
        super(ScalarObjective, self).__init__()
        self.members = members
        self.scale = scale
        self.bestEnergy = 1e99
        self.ncalls = 0
        
        # If no useKeys preference, use all available fields
        if len(self.members) == 0:
            self.members = couplings.fields.keys()

        terms = {} # dictionary
        for pairKey, pairDiff in couplings.fieldDiffs.iteritems():
            # verify that both fields are members of specified groups
            upperKey, lowerKey = pairKey
            if upperKey not in self.members: continue
            if lowerKey not in self.members: continue
            # determine index of each key is members
            upperIndex = self.members.index(upperKey)
            lowerIndex = self.members.index(lowerKey)
            
            # If initial offsets were provided for fields, apply them here
            initUpperOffset = 0.
            initLowerOffset = 0.
            if upperKey in initOffsets:
                initUpperOffset = initOffsets[upperKey]
            if lowerKey in initOffsets:
                initLowerOffset = initOffsets[lowerKey]
            adjustedPairDiff = pairDiff - initUpperOffset + initLowerOffset
            
            # Now we can confidently pack a coupling term for the terms dictionary
            term = {'delta': adjustedPairDiff,
                    'delta_sigma': couplings.fieldDiffSigmas[pairKey],
                    'delta_area': couplings.fieldDiffAreas[pairKey],
                    'upper_index': upperIndex,
                    'lower_index': lowerIndex,
                    'weight': 1. / couplings.fieldDiffSigmas[pairKey] ** 2.}
            terms[pairKey] = term

        # Repack terms dictionary into numpy arrays
        self.nTerms = len(terms.keys())
        tUpIndex = numpy.zeros(self.nTerms, dtype=DTYPE_I)
        tLoIndex = numpy.zeros(self.nTerms, dtype=DTYPE_I)
        tDelta = numpy.zeros(self.nTerms, dtype=DTYPE_F)
        tDeltaSigma = numpy.zeros(self.nTerms, dtype=DTYPE_F)
        tDeltaArea = numpy.zeros(self.nTerms, dtype=DTYPE_F)
        tWeight = numpy.zeros(self.nTerms, dtype=DTYPE_F)
        for i, (pairKey, t) in enumerate(terms.iteritems()):
            tUpIndex[i] = t['upper_index']
            tLoIndex[i] = t['lower_index']
            tDelta[i] = t['delta']
            tDeltaSigma[i] = t['delta_sigma']
            tDeltaArea[i] = t['delta_area']
            tWeight[i] = t['weight']

        # Recast the numpy buffer types for efficient access
        # see http://groups.google.com/group/cython-users/browse_thread/thread/bc55e7b4bcbfb3d4
        if not tUpIndex.flags['C_CONTIGUOUS']:
            tUpIndex = tUpIndex.copy()
        self.tUpIndex = tUpIndex
        self.tUpIndex_p = <DTYPE_UI_t*>self.tUpIndex.data

        if not tLoIndex.flags['C_CONTIGUOUS']:
            tLoIndex = tLoIndex.copy()
        self.tLoIndex = tLoIndex
        self.tLoIndex_p = <DTYPE_LI_t*>self.tLoIndex.data

        if not tDelta.flags['C_CONTIGUOUS']:
            tDelta = tDelta.copy()
        self.tDelta = tDelta
        self.tDelta_p = <DTYPE_D_t*>self.tDelta.data

        if not tDeltaSigma.flags['C_CONTIGUOUS']:
            tDeltaSigma = tDeltaSigma.copy()
        self.tDeltaSigma = tDeltaSigma
        self.tDeltaSigma_p = <DTYPE_DS_t*>self.tDeltaSigma.data

        if not tDeltaArea.flags['C_CONTIGUOUS']:
             tDeltaArea = tDeltaArea.copy()
        self.tDeltaArea = tDeltaArea
        self.tDeltaArea_p = <DTYPE_DA_t*>self.tDeltaArea.data

        if not tWeight.flags['C_CONTIGUOUS']:
            tWeight = tWeight.copy()
        self.tWeight = tWeight
        self.tWeight_p = <DTYPE_W_t*>self.tWeight.data

        self.bestOffsets = numpy.zeros(len(self.members))
    
    cpdef int get_ndim(self) except *:
        """:return: number of dimensions (equal to number of images)."""
        return len(self.members)

    cpdef double compute(self, numpy.ndarray[DTYPE_F_t, ndim=1] offsets) except *:
        """Compute the objective function, given `offsets`. If the offsets
        present a new minimum, those offsets are saved.

        .. TODO:: check if scaling modifies the offsets array *outside*
            of this method. That is, do I need to copy offsets before I
            scale it?

        :param offsets: numpy array; ordered according to self.members
        :return: objective sum
        """
        self.ncalls += 1
        # De-scale the offsets
        #cdef numpy.ndarray offsets = offsets.copy() / self.scale
        offsets = offsets / self.scale
        
        cdef DTYPE_F_t energy = 0.
        cdef INDEX_t i
        cdef DTYPE_F_t levelDiff
        cdef DTYPE_I_t iU # upper index
        cdef DTYPE_I_t iL # lower index
        for i in xrange(self.nTerms):
            iU = self.tUpIndex_p[i]
            iL = self.tLoIndex_p[i]

            levelDiff = self.tWeight_p[i] * (self.tDelta_p[i]
                    - offsets[iU] + offsets[iL])
            energy = energy + levelDiff * levelDiff
        if energy < self.bestEnergy:
            self.bestEnergy = energy
            self.bestOffsets = offsets.copy()
        
        # Return the energy
        return energy
    
    cpdef dict get_best_offsets(self):
        """:return: dictionary of the *best* offset for each image, given
            the history of objective function calls."""
        fieldOffsets = {}
        for fieldname, offset in zip(self.members, self.bestOffsets):
            fieldOffsets[fieldname] = offset
        return fieldOffsets
