import os
import numpy as np
from blockdb import BlockDB
from mosaicdb import AltPlanarMosaicFactory
from mosaicdb import MosaicDB
from planarmultisimplex import PlanarMultiStartSimplex
import math

import matplotlib.pyplot as plt

def main():
    #vary_starts_4_fields()
    fourField = FourField()
    fourField.starts_vs_best_objf()

def vary_starts_4_fields():
    """Solve four fields, but with variable numbers of stars."""
    nStarts = [1,10,100,1000,10000,100000,1000000]
    #nStarts = [1000]
    fields = ["M31-1","M31-2","M31-3","M31-4"]
    for n in nStarts:
        mosaicName = "mock_4_field_nstart_%i"%n
        solve_mosaic(mosaicName, n, fields, "vary_starts_4_fields")

def solve_mosaic(name, nStarts, fields, seriesName):
    blockDB = BlockDB(dbname="skyoffsets", cname="mock_blocks")
    mosaicDir = "skyoffsets/mock_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    initLevelOffsets = get_scalar_result()
    mosaicDB = AltPlanarMosaicFactory(dbname="m31", cname="mosaics")
    mosaicDB.init_scalar_offsets(initLevelOffsets)
    mosaicDB.set_plane_distributions(5e-9, 5e-13, 0.1)
    mosaicDB.make(blockDB, "J", mosaicDir, name,
                solverDBName="skyoffsets",
                nRuns=nStarts,
                resetCouplings=True,
                mosaicMeta={"FILTER": "J", "INSTR": "WIRCam", "type": "planar",
                    "mock_series": seriesName},
                fieldnames=fields,
                excludeFields=["M31-44","M31-45","M31-46"])
    

def get_scalar_result():
    """Get best_fit scalar offsets for the mock planar mosaic."""
    mosaicname = "mock_plane_scalar_J"
    mosaicDB = MosaicDB(dbname="m31", cname="mosaics")
    doc = mosaicDB.get_mosaic_doc(mosaicname)
    return doc['offsets']

class FourField(object):
    """Convergence tests for four field experiment"""
    def __init__(self):
        super(FourField, self).__init__()
        self.mosaicDB = AltPlanarMosaicFactory(dbname="m31", cname="mosaics")
        self.blockDB = BlockDB(dbname="skyoffsets", cname="mock_blocks")
        
        self.nStarts = np.array([1,10,100,1000,10000,100000])
        self.mosaicNames = ["mock_4_field_nstart_%i"%i for i in self.nStarts]

        self.plotDir = os.path.join("skyoffsets","mockplaneexp")
        if os.path.exists(self.plotDir) is False:
            os.makedirs(self.plotDir)
    
    def starts_vs_best_objf(self):
        """Plots number of starts against objf and other goodness of fit
        statistics
        """
        fopts = self.get_best_fopts()
        rmmin, rmmed, rmmax, rcmin, rcmed, rcmax = self.get_residuals_vs_starts()

        fig = plt.figure(figsize=(3,6))
        fig.subplots_adjust(left=0.25, bottom=None, right=None, top=None, wspace=None, hspace=None)
        axSlope = fig.add_subplot(311)
        axLevel = fig.add_subplot(312)
        axFopt = fig.add_subplot(313)

        axSlope.fill_between(np.log10(self.nStarts), rmmax, y2=rmmin,
                facecolor='y',alpha=0.5)
        axSlope.plot(np.log10(self.nStarts), rmmed, 'ok')
        axLevel.fill_between(np.log10(self.nStarts), rcmax, y2=rcmin,
                facecolor='y', alpha=0.5)
        axLevel.plot(np.log10(self.nStarts), rcmed, 'ok')

        axFopt.semilogy(np.log10(self.nStarts), fopts, 'ok')

        axFopt.set_xlabel(r"$\log_{10}N_\mathrm{starts}$")
        axSlope.set_ylabel("Residual Slope")
        axLevel.set_ylabel("Residual Level")
        axFopt.set_ylabel(r"$\mathcal{F}_\mathrm{opt}$")

        axSlope.set_xlim(0.,5)
        axLevel.set_xlim(0.,5)
        axFopt.set_xlim(0.,5.)

        for label in axSlope.xaxis.get_majorticklabels(): label.set_visible(False)
        for label in axLevel.xaxis.get_majorticklabels(): label.set_visible(False)

        fig.savefig("skyoffsets/mockplaneexp/mockplaneexp.pdf", format="pdf")


    def get_residuals_vs_starts(self):
        """docstring for get_residual_slopes_vs_starts"""
        worstResidualSlope = []
        bestResidualSlope = []
        medianResidualSlope = []

        worstResidualLevel = []
        bestResidualLevel = []
        medianResidualLevel = []
        
        for mosaicName in self.mosaicNames:
            rec = self.mosaicDB.collection.find_one({"_id": mosaicName})
            diffNames = rec['couplings']['diffs'].keys()
            residualSlopes = []
            residualLevels = []
            for diffName in diffNames:
                upperImage, lowerImage = diffName.split("*")
                diff = rec['couplings']['diffs'][diffName]
                lowerTrans = rec['couplings']['lower_trans'][diffName]
                upperTrans = rec['couplings']['upper_trans'][diffName]
                pU = rec['offsets'][upperImage]
                pL = rec['offsets'][lowerImage]
                blockC = self.blockDB.collection
                truePU = blockC.find_one({"_id": upperImage})['true_plane']
                truePL = blockC.find_one({"_id": lowerImage})['true_plane']
                
                mx = diff[0] - pU[0] + pL[0]
                my = diff[1] - pU[1] + pL[1]
                c = diff[2] \
                    - (pU[2] + upperTrans[0]*pU[0] + upperTrans[1]*pU[1]) \
                    + (pL[2] + lowerTrans[0]*pL[0] + lowerTrans[1]*pL[1])
                
                residualSlopes.append(math.fabs(mx))
                residualSlopes.append(math.fabs(my))
                residualLevels.append(math.fabs(c))

            residualSlopes = np.array(residualSlopes)
            residualLevels = np.array(residualLevels)
            worstResidualSlope.append(residualSlopes.max())
            worstResidualLevel.append(residualLevels.max())
            medianResidualSlope.append(np.median(residualSlopes))
            medianResidualLevel.append(np.median(residualLevels))
            bestResidualSlope.append(residualSlopes.min())
            bestResidualLevel.append(residualLevels.min())
        return worstResidualSlope, medianResidualSlope, bestResidualSlope, \
            worstResidualLevel, medianResidualLevel, bestResidualLevel

    def get_best_fopts(self):
        bestFopts = []
        for mosaicName in self.mosaicNames:
            rec = self.mosaicDB.collection.find_one({"_id": mosaicName})
            solverDBName = rec['solver_db']
            solverCName = rec['solver_cname']
            solver = PlanarMultiStartSimplex(dbname=solverDBName,
                    cname=solverCName)
            recs = solver.collection.find({}, ['best_fopt'])
            fopts = []
            for rec in recs:
                fopt = rec['best_fopt']
                fopts.append(fopt)
            bestFopts.append(min(fopts))
        return bestFopts


if __name__ == '__main__':
    main()
