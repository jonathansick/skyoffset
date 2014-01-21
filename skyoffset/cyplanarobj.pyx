"""Planar sky offset objective function (cython version)"""
import numpy
cimport numpy
import math

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
ctypedef numpy.float_t DTYPE_UT_t
ctypedef numpy.float_t DTYPE_LT_t
ctypedef numpy.float_t DTYPE_HW_t
ctypedef numpy.float_t DTYPE_HH_t
ctypedef numpy.float_t DTYPE_BP_t
ctypedef Py_ssize_t INDEX_t

cdef class PlanarObjective:
    """Objective function for planar sky offsets.

    A plane is described by a tuple of mx, my, constant

    :param couplings: difftools CoupledPlanes object
    :param members: limit objective function to be sensitive to a limited
        set of images. Leave as None if all images in the `couplings` should
        be considered.
    :param levelScale: All physical slopes are *multiplied* by this
        value before being inserted into the simplex.
    :param slopeScale: All slopes are also multiplied by this value so
        that their variation is similar to the levels. Thus slopes are scaled
        by both factors of `levelScale` and `slopeScale`.
    """
    cdef list members
    cdef DTYPE_F_t levelScale
    cdef DTYPE_F_t slopeScale
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
    cdef DTYPE_UT_t* tUpTrans_p
    cdef DTYPE_LT_t* tLoTrans_p
    cdef DTYPE_HW_t* tHalfWidth_p
    cdef DTYPE_HH_t* tHalfHeight_p
    # Declare numpy array members
    cdef numpy.ndarray tUpIndex
    cdef numpy.ndarray tLoIndex
    cdef numpy.ndarray tDelta
    cdef numpy.ndarray tDeltaSigma
    cdef numpy.ndarray tDeltaArea
    cdef numpy.ndarray tWeight
    cdef numpy.ndarray tUpTrans
    cdef numpy.ndarray tLoTrans
    cdef numpy.ndarray tHalfWidth
    cdef numpy.ndarray tHalfHeight

    cdef numpy.ndarray bestOffsets


    def __init__(self, object couplings,
            list members=[], float levelScale=1., float slopeScale=1.):
        self.members = members
        self.levelScale = levelScale
        self.slopeScale = slopeScale
        self.bestEnergy = 1e99
        self.ncalls = 0
        
        # If no useKeys preference, use all available fields
        if len(self.members) == 0:
            self.members = couplings.fields.keys()

        terms = {} # dictionary
        for pairKey, pairDiff in couplings.diffPlanes.iteritems():
            # verify that both fields are members of specified groups
            upperKey, lowerKey = pairKey
            if upperKey not in self.members: continue
            if lowerKey not in self.members: continue
            # determine index of each key is members
            upperIndex = self.members.index(upperKey)
            lowerIndex = self.members.index(lowerKey)
            
            # If initial offsets were provided for fields, apply them here
            # TODO or handle this a subclass of CoupledPlanes?
            adjustedPairDiff = pairDiff

            transU = couplings.upperOverlapTrans[pairKey]
            transL = couplings.lowerOverlapTrans[pairKey]
            halfWidth = couplings.overlapShape[pairKey][0] / 2.
            halfHeight = couplings.overlapShape[pairKey][1] / 2.
            
            # Now we can confidently pack a coupling term for the terms dictionary
            term = {'delta': pairDiff,
                    'delta_sigma': couplings.diffSigmas[pairKey],
                    'delta_area': couplings.diffAreas[pairKey],
                    'upper_index': upperIndex,
                    'lower_index': lowerIndex,
                    'weight': 1.,
                    'upper_trans': transU,
                    'lower_trans': transL,
                    'half_width': halfWidth,
                    'half_height': halfHeight}
            terms[pairKey] = term

        # Repack terms dictionary into numpy arrays
        self.nTerms = len(terms.keys())
        tUpIndex = numpy.zeros(self.nTerms, dtype=DTYPE_I)
        tLoIndex = numpy.zeros(self.nTerms, dtype=DTYPE_I)
        tUpTrans = numpy.zeros([self.nTerms,2], dtype=DTYPE_F)
        tLoTrans = numpy.zeros([self.nTerms,2], dtype=DTYPE_F)
        tDelta = numpy.zeros([self.nTerms,3], dtype=DTYPE_F)
        tDeltaSigma = numpy.zeros(self.nTerms, dtype=DTYPE_F)
        tDeltaArea = numpy.zeros(self.nTerms, dtype=DTYPE_F)
        tWeight = numpy.zeros(self.nTerms, dtype=DTYPE_F)
        tHalfWidth = numpy.zeros(self.nTerms, dtype=DTYPE_F)
        tHalfHeight = numpy.zeros(self.nTerms, dtype=DTYPE_F)
        for i, (pairKey, t) in enumerate(terms.iteritems()):
            tUpIndex[i] = t['upper_index']
            tLoIndex[i] = t['lower_index']
            tDelta[i,:] = t['delta']
            tUpTrans[i,:] = t['upper_trans']
            tLoTrans[i,:] = t['lower_trans']
            tDeltaSigma[i] = t['delta_sigma']
            tDeltaArea[i] = t['delta_area']
            tWeight[i] = t['weight']
            tHalfWidth[i] = t['half_width']
            tHalfHeight[i] = t['half_height']

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

        if not tUpTrans.flags['C_CONTIGUOUS']:
            tUpTrans = tUpTrans.copy()
        self.tUpTrans = tUpTrans
        self.tUpTrans_p = <DTYPE_UT_t*>self.tUpTrans.data

        if not tLoTrans.flags['C_CONTIGUOUS']:
            tLoTrans = tLoTrans.copy()
        self.tLoTrans = tLoTrans
        self.tLoTrans_p = <DTYPE_LT_t*>self.tLoTrans.data

        if not tHalfWidth.flags['C_CONTIGUOUS']:
            tHalfWidth = tHalfWidth.copy()
        self.tHalfWidth = tHalfWidth
        self.tHalfWidth_p = <DTYPE_HW_t*>self.tHalfWidth.data

        if not tHalfHeight.flags['C_CONTIGUOUS']:
            tHalfHeight = tHalfHeight.copy()
        self.tHalfHeight = tHalfHeight
        self.tHalfHeight_p = <DTYPE_HH_t*>self.tHalfHeight.data

        self.bestOffsets = numpy.zeros([len(self.members),3], dtype=DTYPE_F)
    
    cpdef int get_ndim(self) except *:
        """:return: number of dimensions (equal to number of images
            times 3 dimensions each)."""
        return len(self.members) * 3
    
    cpdef numpy.ndarray[DTYPE_F_t, ndim=2] randomize_simplex(self, float sigma):
        """Initialize the random simplex, using gaussian-distributed offsets
        and slopes.
        
        `sigma` is the width of this distribution in **physical** level
        units.

        In the compute() method, the simplex is scale according to
        `self.levelScale` and `self.slopeScale`. Thus the ratio of
        slope sigma to level sigma is `self.slopeScale/self.levelScale`.
        """
        nFields = len(self.members)
        cdef numpy.ndarray simplex = sigma*self.levelScale \
                * numpy.random.standard_normal([nFields*3+1, nFields*3])
        return simplex

    cpdef numpy.ndarray[DTYPE_F_t, ndim=2] restart_simplex(self,
            numpy.ndarray[DTYPE_F_t, ndim=1] bestOffsets,
            float offsetSigma):
        """Make a restart simplex, where one vertex is the best offset,
        and the others are randomly distributed about that best offset.
        """
        ndim = self.get_ndim()
        cdef numpy.ndarray simplex = numpy.zeros([ndim+1,ndim], dtype=DTYPE_F)
        simplex[0,:] = bestOffsets[:].copy()
        for i in xrange(1, ndim+1):
            simplex[i,:] = offsetSigma*numpy.random.randn(ndim) \
                    + bestOffsets
        return simplex

    cpdef double compute(self, numpy.ndarray[DTYPE_F_t, ndim=1] offsets) except *:
        """Compute the objective function, given `offsets`. If the offsets
        present a new minimum, those offsets are saved.

        .. TODO:: check if scaling modifies the offsets array *outside*
            of this method. That is, do I need to copy offsets before I
            scale it?

        :param offsets: numpy array; planes in serial format.
            TODO document this
        :return: objective sum
        """
        self.ncalls += 1
        # Make a local copy of the offsets
        cdef numpy.ndarray p = numpy.zeros(len(offsets), dtype=DTYPE_F)
        # De-scale slopes relevtive to the level
        p = offsets / self.levelScale
        p[0::3] = p[0::3] / self.slopeScale
        p[1::3] = p[1::3] / self.slopeScale

        print "suggested offset p", p
        print "nTerms:", self.nTerms

        #offsets = offsets / self.scale
        
        cdef DTYPE_F_t energy = 0.
        cdef INDEX_t i # term index
        cdef DTYPE_F_t levelDiff
        cdef DTYPE_I_t iU # upper index
        cdef DTYPE_I_t iL # lower index

        cdef DTYPE_F_t transUX
        cdef DTYPE_F_t transUY
        cdef DTYPE_F_t transLX
        cdef DTYPE_F_t transLY

        cdef DTYPE_F_t mx # x-slope of offset difference plane
        cdef DTYPE_F_t my # y-slope of offset difference plane
        cdef DTYPE_F_t c # level of offset difference plane
        cdef DTYPE_F_t rf # residual flux of offset difference plane

        cdef int xti # x index to Trans_p array
        cdef int yti # y index to Trans_p array

        for i in xrange(self.nTerms):
            iU = self.tUpIndex_p[i]
            iL = self.tLoIndex_p[i]
            print "iU, iL:", iU, iL
            print "pU", p[3*iU], p[3*iU+1], p[3*iU+2]
            print "pL", p[3*iL], p[3*iL+1], p[3*iL+2]
            print "tDelta:", self.tDelta_p[3*i], self.tDelta_p[3*i+1], self.tDelta_p[3*i+2]
            
            mx = self.tDelta_p[3*i] - p[3*iU] + p[3*iL]
            my = self.tDelta_p[3*i+1] - p[3*iU+1] + p[3*iL+1]
            xti = i*2
            yti = xti+1
            c = self.tDelta_p[3*i+2] \
                - (p[3*iU+2] + p[3*iU]*self.tUpTrans_p[xti] + p[3*iU+1]*self.tUpTrans_p[yti]) \
                + (p[3*iL+2] + p[3*iL]*self.tLoTrans_p[xti] + p[3*iL+1]*self.tLoTrans_p[yti])
            
            print "resid plane:", mx, my, c
            # Compute residual flux of the offset difference plane
            rf = numpy.fabs(mx*math.pow(self.tHalfWidth_p[i]/2.,2.)) \
                    + numpy.fabs(my*math.pow(self.tHalfHeight_p[i]/2.,2.)) \
                    + numpy.fabs(c*4.*self.tHalfWidth_p[i]*self.tHalfHeight_p[i])
            
            cc = self.tDelta_p[3*i+1] - p[3*iU+2] + p[3*iL+2]
            #print "cc", cc

            #energy = energy + self.tWeight_p[i]*rf
            energy = energy + cc**2. + (self.slopeScale*mx)**2. + (self.slopeScale*my)**2.
            #energy = energy + (self.slopeScale*mx)**2. + (self.slopeScale*my)**2.
        print "energy=%.2e"%energy
        print "\t=="
        if energy < self.bestEnergy:
            print "\t==better energy!"
            self.bestEnergy = energy
            self.bestOffsets = offsets.copy() # copy original, unscaled offsets
        
        return energy
    
    cpdef dict get_best_offsets(self):
        """:return: dictionary of the *best* offset for each image, given
            the history of objective function calls. Each offset is a tuple
            indicating the (mx, my, c) of the plane. These offsets are in
            physical (not scaled) units
        """
        fieldOffsets = {}
        # Remember that the self.bestOffsets array packs planes serially
        # and that these must be rescaled to be physical
        for i, fieldname in enumerate(self.members):
            j = i*3
            mx = self.bestOffsets[j] / self.levelScale / self.slopeScale
            my = self.bestOffsets[j+1] / self.levelScale / self.slopeScale
            c = self.bestOffsets[j+2] / self.levelScale
            fieldOffsets[fieldname] = (float(mx),float(my),float(c))
        return fieldOffsets

cdef class AltPlaneObjective:
    """
    Base Objective function for planes where only levels or slopes are allowed
    to be optimized at a time.
    """
    cdef list members
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
    cdef DTYPE_UT_t* tUpTrans_p
    cdef DTYPE_LT_t* tLoTrans_p
    cdef DTYPE_HW_t* tHalfWidth_p
    cdef DTYPE_HH_t* tHalfHeight_p
    # Declare numpy array members
    cdef numpy.ndarray tUpIndex
    cdef numpy.ndarray tLoIndex
    cdef numpy.ndarray tDelta
    cdef numpy.ndarray tDeltaSigma
    cdef numpy.ndarray tDeltaArea
    cdef numpy.ndarray tWeight
    cdef numpy.ndarray tUpTrans
    cdef numpy.ndarray tLoTrans
    cdef numpy.ndarray tHalfWidth
    cdef numpy.ndarray tHalfHeight

    # Baseplane
    cdef DTYPE_BP_t* basePlane_p
    cdef numpy.ndarray basePlane

    cdef numpy.ndarray bestOffsets

    def __init__(self, couplings, dict initPlanes, list members=[]):
        self.members = members
        self.bestEnergy = 1e99
        self.ncalls = 0

        # If no useKeys preference, use all available fields
        if len(self.members) == 0:
            self.members = couplings.fields.keys()
        
        self._init_terms(couplings)
        self._init_planes(initPlanes)
        self.bestOffsets = numpy.zeros(self.get_ndim(), dtype=DTYPE_F)

    def _init_terms(self, couplings):
        """Set up the term arrays for difference planes"""
        self.nTerms = len(couplings.diffPlanes.keys())
        pairKeyLst = []
        diffLst = []
        sigmaLst = []
        uIndexLst = []
        lIndexLst = []
        weightLst = []
        uTransLst = []
        lTransLst = []
        areaLst = []
        hwLst = []
        hhLst = []
        for pairKey, pairDiff in couplings.diffPlanes.iteritems():
            upperKey, lowerKey = pairKey
            if upperKey not in self.members: continue
            if lowerKey not in self.members: continue
            # determine index of each key is members
            upperIndex = self.members.index(upperKey)
            lowerIndex = self.members.index(lowerKey)
            transU = couplings.upperOverlapTrans[pairKey]
            transL = couplings.lowerOverlapTrans[pairKey]
            halfWidth = couplings.overlapShape[pairKey][0] / 2.
            halfHeight = couplings.overlapShape[pairKey][1] / 2.
            diffArea = couplings.diffAreas[pairKey]
            
            pairKeyLst.append(pairKey)
            diffLst.append(pairDiff)
            uIndexLst.append(upperIndex)
            lIndexLst.append(lowerIndex)
            weightLst.append(1.)
            uTransLst.append(transU)
            lTransLst.append(transL)
            hwLst.append(halfWidth)
            hhLst.append(halfHeight)
            areaLst.append(diffArea)

        
        tDelta = numpy.array(diffLst, dtype=DTYPE_F)
        if not tDelta.flags['C_CONTIGUOUS']:
            tDelta = tDelta.copy()
        self.tDelta = tDelta
        self.tDelta_p = <DTYPE_D_t*>self.tDelta.data

        tUpIndex = numpy.array(uIndexLst, dtype=DTYPE_I)
        if not tUpIndex.flags['C_CONTIGUOUS']:
            tUpIndex = tUpIndex.copy()
        self.tUpIndex = tUpIndex
        self.tUpIndex_p = <DTYPE_UI_t*>self.tUpIndex.data

        tLoIndex = numpy.array(lIndexLst, dtype=DTYPE_I)
        if not tLoIndex.flags['C_CONTIGUOUS']:
            tLoIndex = tLoIndex.copy()
        self.tLoIndex = tLoIndex
        self.tLoIndex_p = <DTYPE_LI_t*>self.tLoIndex.data

        tWeight = numpy.array(weightLst, dtype=DTYPE_F)
        if not tWeight.flags['C_CONTIGUOUS']:
            tWeight = tWeight.copy()
        self.tWeight = tWeight
        self.tWeight_p = <DTYPE_W_t*>self.tWeight.data
        
        tUpTrans = numpy.array(uTransLst, dtype=DTYPE_I)
        if not tUpTrans.flags['C_CONTIGUOUS']:
            tUpTrans = tUpTrans.copy()
        self.tUpTrans = tUpTrans
        self.tUpTrans_p = <DTYPE_UT_t*>self.tUpTrans.data
        
        tLoTrans = numpy.array(lTransLst, dtype=DTYPE_I)
        if not tLoTrans.flags['C_CONTIGUOUS']:
            tLoTrans = tLoTrans.copy()
        self.tLoTrans = tLoTrans
        self.tLoTrans_p = <DTYPE_LT_t*>self.tLoTrans.data

        tHalfWidth = numpy.array(hwLst, dtype=DTYPE_I)
        if not tHalfWidth.flags['C_CONTIGUOUS']:
            tHalfWidth = tHalfWidth.copy()
        self.tHalfWidth = tHalfWidth
        self.tHalfWidth_p = <DTYPE_HW_t*>self.tHalfWidth.data

        tHalfHeight = numpy.array(hhLst, dtype=DTYPE_I)
        if not tHalfHeight.flags['C_CONTIGUOUS']:
            tHalfHeight = tHalfHeight.copy()
        self.tHalfHeight = tHalfHeight
        self.tHalfHeight_p = <DTYPE_HH_t*>self.tHalfHeight.data

        tDeltaArea = numpy.array(areaLst, dtype=DTYPE_F)
        if not tDeltaArea.flags['C_CONTIGUOUS']:
            tDeltaArea = tDeltaArea.copy()
        self.tDeltaArea = tDeltaArea
        self.tDeltaArea_p = <DTYPE_DA_t*>self.tDeltaArea.data

    def _init_planes(self, dict planes):
        """Initialize the base planes; these are planes the specifiy the
        fixed slope (or level) information. The objective function will
        override the slope (or level) with the optimized versions.
        """
        nMembers = len(self.members)
        basePlane = numpy.zeros([nMembers,3], dtype=DTYPE_F)

        for i, field in enumerate(self.members):
            plane = planes[field]
            basePlane[i,:] = plane
        #print "basePlane:", basePlane

        if not basePlane.flags['C_CONTIGUOUS']:
            basePlane = basePlane.copy()
        self.basePlane = basePlane
        self.basePlane_p = <DTYPE_BP_t*>self.basePlane.data

    cpdef numpy.ndarray[DTYPE_F_t, ndim=2] restart_simplex(self,
            numpy.ndarray[DTYPE_F_t, ndim=1] bestOffsets,
            float offsetSigma):
        """Make a restart simplex, where one vertex is the best offset,
        and the others are randomly distributed about that best offset.
        """
        ndim = self.get_ndim()
        cdef numpy.ndarray simplex = numpy.zeros([ndim+1,ndim], dtype=DTYPE_F)
        simplex[0,:] = bestOffsets[:].copy()
        for i in xrange(1, ndim+1):
            simplex[i,:] = offsetSigma*numpy.random.randn(ndim) \
                    + bestOffsets
        return simplex


cdef class AltLevelObjective(AltPlaneObjective):
    def __init__(self, couplings, dict initPlanes, list members=[]):
        self.members = members
        self.bestEnergy = 1e99
        self.ncalls = 0

        # If no useKeys preference, use all available fields
        if len(self.members) == 0:
            self.members = couplings.fields.keys()
        
        self._init_terms(couplings)
        self._init_planes(initPlanes)

        # Also set bestOffsets to use the basePlane by default
        self.bestOffsets = numpy.zeros(self.get_ndim(), dtype=DTYPE_F)
        for i, field in enumerate(self.members):
            self.bestOffsets[i] = initPlanes[field][2]
    
    cpdef int get_ndim(self) except *:
        return len(self.members)

    cpdef numpy.ndarray[DTYPE_F_t, ndim=2] randomize_simplex(self,
            float levelSigma):
        """Generate a Gaussian-distributed simplex of level offsets.
        The first simplex uses the levels from initPlanes, the rest
        of the points are 'inflations' about those default planes.
        """
        #print "level objective basePlane"
        #print self.basePlane
        n = self.get_ndim()
        cdef numpy.ndarray baseLevels = numpy.zeros(n, dtype=DTYPE_F) #self.basePlane[2::3].copy()
        for i in xrange(n):
            baseLevels[i] = self.basePlane[i,2]
        cdef numpy.ndarray simplex = levelSigma \
                * numpy.random.standard_normal(size=[n+1,n])
        nBaseLevels = len(baseLevels)
        #print "nBaseLevels", nBaseLevels
        simplex[0,:] = baseLevels
        for j in xrange(1,n+1):
            simplex[j,:] = simplex[j,:] + baseLevels
        #print "level objective random simplex:"
        #print simplex
        return simplex

    cpdef double compute(self, numpy.ndarray[DTYPE_F_t, ndim=1] levels) except *:
        """Compute objective function, using these levels in conjunction with
        the base planes
        """
        self.ncalls += 1
        
        cdef DTYPE_F_t energy = 0.
        cdef DTYPE_F_t mx # x-slope of offset difference plane
        cdef DTYPE_F_t my # y-slope of offset difference plane
        cdef DTYPE_I_t iU # upper index
        cdef DTYPE_I_t iL # lower index
        cdef int xti # x index to Trans_p array
        cdef int yti # y index to Trans_p array

        cdef int i
        for i in xrange(self.nTerms):
            iU = self.tUpIndex_p[i]
            iL = self.tLoIndex_p[i]
            xti = i*2
            yti = xti+1
            
            # Computation of residual plane (mx, my, c)
            mx = self.tDelta_p[3*i] \
                    - self.basePlane_p[3*iU] + self.basePlane_p[3*iL]
            my = self.tDelta_p[3*i+1] \
                    - self.basePlane_p[3*iU+1] + self.basePlane_p[3*iL+1]
            c = self.tDelta_p[3*i+2] \
                    - (levels[iU] + self.basePlane_p[3*iU]*self.tUpTrans[i,0] + self.basePlane_p[3*iU+1]*self.tUpTrans[i,1]) \
                    + (levels[iL] + self.basePlane_p[3*iL]*self.tLoTrans[i,0] + self.basePlane_p[3*iL+1]*self.tLoTrans[i,1])
            rf = self.tWeight_p[i]*self.tDeltaArea_p[i]*c
            energy += rf**2

        if energy < self.bestEnergy:
            self.bestEnergy = energy
            self.bestOffsets = levels.copy()
        return energy

    cpdef dict get_best_offsets(self):
        """Returns optimized *planes*, in a field dictionary"""
        fieldOffsets = {}
        for i, fieldname in enumerate(self.members):
            c = float(self.bestOffsets[i])
            mx = float(self.basePlane[i,0])
            my = float(self.basePlane[i,1])
            fieldOffsets[fieldname] = (mx, my, c)
        #print "level objecive get_best_offsets"
        #print fieldOffsets
        return fieldOffsets


cdef class AltSlopeObjective(AltPlaneObjective):
    def __init__(self, couplings, dict initPlanes, list members=[]):
        self.members = members
        self.bestEnergy = 1e99
        self.ncalls = 0

        # If no useKeys preference, use all available fields
        if len(self.members) == 0:
            self.members = couplings.fields.keys()
        
        self._init_terms(couplings)
        self._init_planes(initPlanes)
        self.bestOffsets = numpy.zeros(self.get_ndim(), dtype=DTYPE_F)
        for i, field in enumerate(self.members):
            self.bestOffsets[2*i] = initPlanes[field][0]
            self.bestOffsets[2*i+1] = initPlanes[field][1]
    
    cpdef int get_ndim(self) except *:
        return len(self.members) * 2

    cpdef numpy.ndarray[DTYPE_F_t, ndim=2] randomize_simplex(self, float slopeSigma):
        """Generate a Gaussian-distributed simplex of level offsets"""
        n = self.get_ndim()
        nFields = len(self.members)
        cdef numpy.ndarray baseSlopes = numpy.zeros(n, dtype=DTYPE_F)
        for i in xrange(nFields):
            baseSlopes[2*i] = self.basePlane[i,0]
            baseSlopes[2*i+1] = self.basePlane[i,1]
        cdef numpy.ndarray simplex = slopeSigma \
                * numpy.random.standard_normal(size=[n+1,n])
        #print "slope simplex:"
        #print simplex
        simplex[0,:] = baseSlopes
        for j in xrange(1,n+1):
            simplex[j,:] = simplex[j,:] + baseSlopes
        return simplex

    cpdef double compute_old(self, numpy.ndarray[DTYPE_F_t, ndim=1] s) except *:
        """Compute objective function, using these slopes in conjunction with
        the base plane levels.
        """
        self.ncalls += 1
        
        cdef DTYPE_F_t energy = 0.
        cdef DTYPE_F_t mx # x-slope of offset difference plane
        cdef DTYPE_F_t my # y-slope of offset difference plane
        cdef DTYPE_I_t iU # upper index
        cdef DTYPE_I_t iL # lower index
        cdef int xti # x index to Trans_p array
        cdef int yti # y index to Trans_p array

        cdef int i
        for i in xrange(self.nTerms):
            iU = self.tUpIndex_p[i]
            iL = self.tLoIndex_p[i]
            xti = i*2
            yti = xti+1
            
            # Computation of residual plane (mx, my, c)
            mx = self.tDelta_p[3*i] - s[2*iU] + s[2*iL]
            my = self.tDelta_p[3*i+1] - s[2*iU+1] + s[2*iL+1]

            c = self.tDelta_p[3*i+2] \
                    - (self.basePlane_p[3*iU+2] + s[2*iU]*self.tUpTrans[i,0] + s[2*iU+1]*self.tUpTrans[i,1]) \
                    + (self.basePlane_p[3*iL+2] + s[2*iL]*self.tLoTrans[i,0] + s[2*iL+1]*self.tLoTrans[i,1])
            rf = self.tWeight_p[i]*self.tDeltaArea_p[i]*c
            energy += rf**2

        if energy < self.bestEnergy:
            self.bestEnergy = energy
            self.bestOffsets = s.copy()
        return energy

    cpdef double compute(self, numpy.ndarray[DTYPE_F_t, ndim=1] s) except *:
        """Compute objective function, using these slopes in conjunction with
        the base plane levels.
        """
        self.ncalls += 1
        
        cdef DTYPE_F_t energy = 0.
        cdef DTYPE_F_t mx # x-slope of offset difference plane
        cdef DTYPE_F_t my # y-slope of offset difference plane
        cdef DTYPE_I_t iU # upper index
        cdef DTYPE_I_t iL # lower index
        cdef int xti # x index to Trans_p array
        cdef int yti # y index to Trans_p array

        cdef int i
        for i in xrange(self.nTerms):
            iU = self.tUpIndex_p[i]
            iL = self.tLoIndex_p[i]
            xti = i*2
            yti = xti+1
            
            # Computation of residual plane (mx, my, c)
            mx = self.tDelta_p[3*i] - s[2*iU] + s[2*iL]
            my = self.tDelta_p[3*i+1] - s[2*iU+1] + s[2*iL+1]

            #c = self.tDelta_p[3*i+2] \
            #        - (self.basePlane_p[3*iU+2] + s[2*iU]*self.tUpTrans[i,0] + s[2*iU+1]*self.tUpTrans[i,1]) \
            #        + (self.basePlane_p[3*iL+2] + s[2*iL]*self.tLoTrans[i,0] + s[2*iL+1]*self.tLoTrans[i,1])
            #rf = self.tWeight_p[i]*self.tDeltaArea_p[i]*c
            energy += rf**2

            rf = mx**2. + my**2.
            energy += rf
            
        if energy < self.bestEnergy:
            self.bestEnergy = energy
            self.bestOffsets = s.copy()
        return energy

    cpdef dict get_best_offsets(self):
        """Returns optimized *planes*, in a field dictionary"""
        fieldOffsets = {}
        #print "slope obj get_best_offsets"
        #print self.bestOffsets
        for i, fieldname in enumerate(self.members):
            c = float(self.basePlane[i,2])
            mx = float(self.bestOffsets[2*i])
            my = float(self.bestOffsets[2*i+1])
            fieldOffsets[fieldname] = (mx, my, c)
        return fieldOffsets


