#!/usr/bin/env python
# encoding: utf-8
"""
For Computing residual fluxes in the couplings of a mosaic.

History
-------
2011-09-18 - Created by Jonathan Sick

"""


import math

from difftools import Couplings  # Scalar couplings


class ScalarResidualFlux(object):
    """Residual flux in a scalar-fit mosaic"""
    def __init__(self, mosaicdb, mosaicName):
        super(ScalarResidualFlux, self).__init__()
        self.mosaicName = mosaicName
        mosaicDoc = mosaicdb.collection.find_one({"_id": mosaicName})
        self.offsets = mosaicDoc['offsets']
        self.couplings = Couplings.load_doc(mosaicDoc['couplings'])

    def residual_fluxes(self):
        """Returns a dictioanry of pairKey: residual flux
        (residual diff times area in pixels).
        """
        residualFluxes = {}
        for pairKey, diff in self.couplings.fieldDiffs.iteritems():
            area = self.couplings.fieldDiffAreas[pairKey]
            upperField, lowerField = pairKey
            residualFluxes[pairKey] = math.fabs(diff - self.offsets[upperField]
                + self.offsets[lowerField]) * area
        return residualFluxes

    def residual_levels(self):
        """Returns the residual level difference."""
        residualLevels = {}
        for pairKey, diff in self.couplings.fieldDiffs.iteritems():
            upperField, lowerField = pairKey
            residualLevels[pairKey] = math.fabs(diff - self.offsets[upperField]
                + self.offsets[lowerField])
        return residualLevels

    def residual_rel_sigma(self):
        """The residual level relative to the coupling sigma."""
        relResiduals = {}
        for pairKey, diff in self.couplings.fieldDiffs.iteritems():
            upperField, lowerField = pairKey
            sigma = self.couplings.fieldDiffSigmas[pairKey]
            relResiduals[pairKey] = math.fabs(diff - self.offsets[upperField]
                + self.offsets[lowerField]) / sigma
        return relResiduals

    def residual_rel_sblevel(self):
        """Returns ratios of residual level difference / mean intensity level
        in block.
        """
        relResiduals = {}
        for pairKey, diff in self.couplings.fieldDiffs.iteritems():
            upperField, lowerField = pairKey
            fieldLevel = self.couplings.fieldLevels[pairKey]
            relResiduals[pairKey] = math.fabs(diff - self.offsets[upperField]
                + self.offsets[lowerField]) / math.fabs(fieldLevel)
        return relResiduals

    def coupling_sigmas(self):
        """The standard deviations of image overlaps."""
        return self.coupligns.fieldDiffSigmas
