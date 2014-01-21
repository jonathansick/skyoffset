"""Scalar sky offset objective function (pure python version)"""

class ScalarObjective(object):
    """Objective function for scalar sky offsets.
    :param couplings: difftools couplings object
    :param initOffsets: optional dictionary of offsets, keyed by image key.
        Any offset optimized will be **relative** to these initial offsets.
        That is, these offsets are *corrections* to information in the couplings
        object.
    :param members: limit objective function to be sensitive to a limited
        set of images. Leave as None if all images in the `couplings` should
        be considered.
    :param scale: number that physical offsets are multipled by, so that
        the optimization can operate on values closer to unity.
    """
    def __init__(self, couplings, initOffsets={}, members=None, scale=1.):
        super(ScalarObjective, self).__init__()
        self.couplings = couplings
        self.members = members
        self.scale = scale
        self.bestEnergy = 1e99
        self.ncalls = 0
        
        # If no useKeys preference, use all available fields
        if self.members == None:
            self.members = self.couplings.fields.keys()

        self.terms = {} # dictionary
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
                    'delta_area': couplings.fieldDiffSigmas[pairKey],
                    'upper_index': upperIndex,
                    'lower_index': lowerIndex,
                    'weight': 1.}
            self.terms[pairKey] = term
    
    def get_ndim(self):
        """:return: number of dimensions (equal to number of images)."""
        return len(self.members)

    def compute(self, offsets):
        """Compute the objective function, given `offsets`. If the offsets
        present a new minimum, those offsets are saved.
        :param offsets: numpy array; ordered according to self.members
        :return: objective sum
        """
        # De-scale the offsets
        offsets = offsets.copy() / self.scale
        
        energy = 0.
        for (fieldU, fieldL), t in self.terms.iteritems():
            iU = t['upper_index']
            iL = t['lower_index']
            levelDiff = t['weight'] * (t['delta'] - offsets[iU] + offsets[iL])
            energy += levelDiff*levelDiff
        if energy < self.bestEnergy:
            self.bestEnergy = energy
            self.bestOffsets = offsets.copy()
        self.ncalls += 1
        # Return the energy
        return energy
    
    def get_best_offsets(self):
        """:return: dictionary of the *best* offset for each image, given
            the history of objective function calls."""
        fieldOffsets = {}
        for fieldname, offset in zip(self.members, self.bestOffsets):
            fieldOffsets[fieldname] = offset
        return fieldOffsets

