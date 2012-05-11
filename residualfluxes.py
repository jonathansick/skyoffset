#!/usr/bin/env python
# encoding: utf-8
"""
For Computing residual fluxes in the couplings of a mosaic.

History
-------
2011-09-18 - Created by Jonathan Sick

"""

__all__ = ['']

import numpy as np
import math

from mosaicdb import MosaicDB
from difftools import Couplings # Scalar couplings
from difftools import CoupledPlanes # Planar couplings
from multisimplex import SimplexScalarOffsetSolver

def main():
    pass

class ScalarResidualFlux(object):
    """Residual flux in a scalar-fit mosaic"""
    def __init__(self, mosaicName):
        super(ScalarResidualFlux, self).__init__()
        self.mosaicName = mosaicName
        mosaicdb = MosaicDB()
        mosaicDoc = mosaicdb.collection.find_one({"_id": mosaicName})
        solverCName = mosaicDoc['solver_cname']
        solverDBName = mosaicDoc['solver_db']
        
        solver = SimplexScalarOffsetSolver(dbname=solverDBName,
                cname=solverCName)
        self.offsets = solver.find_best_offsets()

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

    def residual_levels_sb(self, pixScale=0.3):
        """Returns the residual level difference in mag/arcsec^2."""
        residualLevels = self.residual_levels()
        residualSBs = {}
        for pairKey, level in residualLevels.iteritems():
            residualSBs[pairKey] = -2.5*math.log10(level/pixScale**2.)
        return residualSBs

    def relative_residuals(self):
        """The residual level relative to the coupling sigma."""
        relResiduals = {}
        for pairKey, diff in self.couplings.fieldDiffs.iteritems():
            upperField, lowerField = pairKey
            sigma = self.couplings.fieldDiffSigmas[pairKey]
            relResiduals[pairKey] = math.fabs(diff - self.offsets[upperField]
                + self.offsets[lowerField]) / sigma
        return relResiduals

    def coupling_sigmas(self):
        """docstring for coupling_sigmas"""
        return self.coupligns.fieldDiffSigmas


if __name__ == '__main__':
    main()


