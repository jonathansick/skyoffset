import pyfits
import os
import numpy as np
import math

from blockdb import BlockDB
from offsettools import make_plane_image
from mosaicdb import PlanarMosaicFactory
from mosaicdb import AltPlanarMosaicFactory
from mosaicdb import MosaicDB

def main():
    #make_mock_blocks()
    #create_couplings()
    #solve_scalar()
    #solve_mosaic()
    alt_solve_mosaic()
    #analyze_diff_planes()

def make_mock_blocks():
    """Make a mock data set by using the WCS footprint of real J band fields,
    but whose data is purely random planes offset from flat. These planes
    are recorded in the BlockDB (which does not overwrite the true BlockDB)"""
    realBlockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mockBlockDB = MockBlockDB(dbname="skyoffsets", cname="mock_blocks")
    mockBlockDB.create_blocks(realBlockDB, "skyoffsets/mock_blocks")

def create_couplings():
    """Create CoupledPlanes to test theory of transformed and subtracted planes."""
    blockDB = BlockDB(dbname="skyoffsets", cname="mock_blocks")
    mosaicDir = "skyoffsets/mock_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    band = "J"
    mosaicDB = PlanarMosaicFactory(dbname="m31", cname="mosaics")
    mosaicName = "mock_%s"%band
    blockDocs = blockDB.find_blocks({"FILTER":band,
        "field":{"$in":["M31-1","M31-2","M31-3","M31-4"]}})
    mosaicDB.workDir = os.path.join(mosaicDir, mosaicName)
    mosaicDB._make_couplings(mosaicName, blockDocs)
    #mosaicDB.make(blockDB, band, mosaicDir, mosaicname,
    #        solverDBName="skyoffsets",
    #        nRuns=1,
    #        resetCouplings=True,
    #        mosaicMeta={"FILTER": band, "INSTR": "WIRCam", "type": "planar"},
    #        fieldnames=includeFields,
    #        excludeFields=["M31-44","M31-45","M31-46"])

def solve_scalar():
    """Solve scalar offsets for the mock plane mosaic"""
    blockDB = BlockDB(dbname="skyoffsets", cname="mock_blocks")
    mosaicDir = "skyoffsets/mock_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    band = "J"
    mosaicname = "mock_plane_scalar_%s"%band
    mosaicDB = MosaicDB(dbname="m31", cname="mosaics")
    fieldnames = None
    mosaicDB.make(blockDB, band, mosaicDir, mosaicname,
                solverDBName="skyoffsets",
                nRuns=1000,
                resetCouplings=True,
                mosaicMeta={"FILTER": band, "INSTR": "WIRCam", "type": "scalar"},
                fieldnames=fieldnames,
                excludeFields=["M31-44","M31-45","M31-46"])

def get_scalar_result():
    """Get best_fit scalar offsets for the mock planar mosaic."""
    mosaicname = "mock_plane_scalar_J"
    mosaicDB = MosaicDB(dbname="m31", cname="mosaics")
    doc = mosaicDB.get_mosaic_doc(mosaicname)
    return doc['offsets']

def solve_mosaic():
    """docstring for solve_mosaic"""
    blockDB = BlockDB(dbname="skyoffsets", cname="mock_blocks")
    mosaicDir = "skyoffsets/mock_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    band = "J"
    mosaicDB = PlanarMosaicFactory(dbname="m31", cname="mosaics")
    mosaicName = "mock_%s"%band
    includeFields = ["M31-1","M31-2","M31-3","M31-4"]
    #includeFields = ["M31-1","M31-2"]
    mosaicDB.make(blockDB, band, mosaicDir, mosaicName,
            solverDBName="skyoffsets",
            nRuns=1000,
            resetCouplings=True,
            mosaicMeta={"FILTER": band, "INSTR": "WIRCam", "type": "planar"},
            fieldnames=includeFields,
            excludeFields=["M31-44","M31-45","M31-46"])

def alt_solve_mosaic():
    """docstring for alt_solve_mosaic"""
    blockDB = BlockDB(dbname="skyoffsets", cname="mock_blocks")
    mosaicDir = "skyoffsets/mock_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    bands = ["J"]
    #includeFields = ["M31-1","M31-2","M31-3","M31-4"]
    includeFields = ["M31-1","M31-2","M31-3","M31-4"]
    initLevelOffsets = get_scalar_result()
    for band in bands:
        mosaicDB = AltPlanarMosaicFactory(dbname="m31", cname="mosaics")
        mosaicname = "mock_alt_%s"%band
        mosaicDB.init_scalar_offsets(initLevelOffsets)
        mosaicDB.set_plane_distributions(5e-9, 5e-13, 0.1)
        mosaicDB.make(blockDB, band, mosaicDir, mosaicname,
                solverDBName="skyoffsets",
                nRuns=10000, # TODO CHANGEME
                resetCouplings=True,
                mosaicMeta={"FILTER": band, "INSTR": "WIRCam", "type": "planar"},
                fieldnames=includeFields,
                excludeFields=["M31-44","M31-45","M31-46"])

class MockBlockDB(BlockDB):
    """DB for blocks that are pure and determinately-known planes."""
    def __init__(self, **kwargs):
        super(MockBlockDB, self).__init__(**kwargs)
    
    def create_blocks(self, realBlockDB, blockDir):
        """Copy the frames from real blocks, and replace image data with known
        planes. The true planes are recorded in the `true_plane` key.
        """
        # Reset collection
        self.collection.drop()
        self.collection = self.db[self.cname]

        if os.path.exists(blockDir) is False: os.makedirs(blockDir)
        recs = realBlockDB.collection.find({})
        for rec in recs:
            doc = rec
            origImagePath = doc['image_path']
            mockImagePath, mockPlane = self._make_mock_image(origImagePath,
                    blockDir)
            doc['image_path'] = mockImagePath
            doc['true_plane'] = mockPlane
            self.collection.insert(doc)

    def _make_mock_image(self, origImagePath, blockDir):
        mockPlane = self._make_mock_plane()
        originalFITS = pyfits.open(origImagePath)
        shape = originalFITS[0].data.shape
        header = originalFITS[0].header
        planeImage = make_plane_image(shape, mockPlane).astype(np.float32)
        basename = os.path.basename(origImagePath)
        planePath = os.path.join(blockDir, basename)
        pyfits.writeto(planePath, planeImage, header, clobber=True)
        return planePath, mockPlane

    def _make_mock_plane(self):
        """Randomly produce a plane with realistic amplitudes"""
        slopeSigma = 5e-13
        levelSigma = 5e-9
        mx = slopeSigma*np.random.randn()
        my = slopeSigma*np.random.randn()
        c = levelSigma*np.random.randn()
        return mx, my, c

def analyze_diff_planes():
    """Determine if the computed difference planes are what I expect."""
    blockDB = MockBlockDB(dbname="skyoffsets", cname="mock_blocks")
    band = "J"
    mosaicDB = PlanarMosaicFactory(dbname="m31", cname="mosaics")
    mosaicName = "mock_alt_%s"%band
    rec = mosaicDB.collection.find_one({"_id": mosaicName})
    diffNames = rec['couplings']['diffs'].keys()
    print "Differences:", diffNames
    diffNames.sort()
    for diffName in diffNames:
        upperImage, lowerImage = diffName.split("*")
        print "== %s - %s" % (upperImage, lowerImage)
        diff = rec['couplings']['diffs'][diffName]
        lowerTrans = rec['couplings']['lower_trans'][diffName]
        upperTrans = rec['couplings']['upper_trans'][diffName]
        blockC = blockDB.collection
        pU = blockC.find_one({"_id": upperImage})['true_plane']
        pL = blockC.find_one({"_id": lowerImage})['true_plane']
        print "Upper plane:", upperImage, pU
        print "Lower plane:", lowerImage, pL
        
        mx = diff[0] - pU[0] + pL[0]
        my = diff[1] - pU[1] + pL[1]
        c = diff[2] \
            - (pU[2] + upperTrans[0]*pU[0] + upperTrans[1]*pU[1]) \
            + (pL[2] + lowerTrans[0]*pL[0] + lowerTrans[1]*pL[1])
    
        print "\tOriginal Difference:", diff[0], diff[1], diff[2]
        print "\tMinimized Difference", mx, my, c


if __name__ == '__main__':
    main()
