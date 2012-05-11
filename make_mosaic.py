"""Controller script for making mosaics from resampled images"""
import os

from andpipe.imagelog import WIRLog
from resampledsets import ResampledDataset
from stackdb import StackDB
from blockdb import BlockDB
from mosaicdb import MosaicDB
from mosaicdb import PlanarMosaicFactory
from mosaicdb import AltPlanarMosaicFactory

def main():
    """Functions to make mosaics from WIRCam images using sky offsets."""
    imageLog = WIRLog()
    allFields = imageLog.find_unique("OBJECT", selector={"TYPE":"sci"})
    allBands = ["J","Ks"]
    #resampledSet = ResampledDataset(imageLog, dbname="m31", cname="wircam_resampled")
    
    #print "Making stacks"
    #for field in allFields:
    #    for band in allBands:
    #        make_stacks(imageLog, resampledSet, field, band)
    
    #print "Making blocks"
    #allBands = ["J"]
    #make_blocks(allFields, allBands)

    #print "Making scalar mosaics"
    #make_scalar_mosaics(allBands)

    print "Making planar mosaics"
    #make_planar_mosaics(['J'], includeFields=["M31-1","M31-2","M31-3","M31-4"])
    #make_planar_mosaics(['J'], includeFields=["M31-1","M31-2"])
    #make_alt_planar_mosaics(['J'], includeFields=['M31-1','M31-2'])
    #make_alt_planar_mosaics(['J'], includeFields=['M31-1','M31-2',"M31-3","M31-4",
    #    "M31-7","M31-8"])

    #fields = ['M31-1',"M31-2","M31-3","M31-4"]
    #fields = ['M31-1',"M31-2","M31-3","M31-4","M31-7","M31-8",
    #        'M31-32','M31-33','M31-34']
    
    # Whole galaxy alt-plane mosaic
    restrictedFields = ["M31-%i"%i for i in range(1,28)] \
        + ["M31-%i"%i for i in range(32,43)]
    #make_alt_planar_mosaics(['J'], includeFields=restrictedFields, nThreads=7,
    #        reset=False)
    
    # Make new mosaic based on results from `make_alt_planar_mosaics`
    plot_mosaic(restrictedFields)
    
    # Run with fields in the middle of the galaxy
    #midgalaxy_alt_plane_mosaic_experiment(nThreads=4, nTrials=1000, reset=False)

def make_stacks(imageLog, resampledSet, field, band):
    workDir = "skyoffsets/stacks"
    if os.path.exists(workDir) is False: os.makedirs(workDir)
    stackDB = StackDB(dbname="m31", cname="wircam_stacks")
    for ext in [1,2,3,4]:
        stackDir = os.path.join(workDir, "%s_%s_%i"%(field,band,ext))
        if os.path.exists(stackDir):
            continue
        else:
            print "working on", stackDir
            stackDB.stack(imageLog, resampledSet, field, band, ext, workDir)

def make_blocks(fields, bands):
    """Make WIRCam blocks."""
    print "Making blocks"
    stackDB = StackDB(dbname="m31", cname="wircam_stacks")
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    blockDir = "skyoffsets/blocks"
    if os.path.exists(blockDir) is False: os.makedirs(blockDir)
    for field in fields:
        for band in bands:
            solvercname = "solver_%s_%s"%(field,band)
            blockDB.make(stackDB, field, band, blockDir, solvercname)

def make_scalar_mosaics(bands):
    """Make WIRCam scalar mosaics, currently excluding the short exposure fields."""
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mosaicDir = "skyoffsets/scalar_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    for band in bands:
        mosaicDB = MosaicDB(dbname="m31", cname="mosaics")
        fieldnames = None
        mosaicname = "scalar_%s"%band
        mosaicDB.make(blockDB, band, mosaicDir, mosaicname,
                solverDBName="skyoffsets",
                nRuns=1000,
                resetCouplings=True,
                mosaicMeta={"FILTER": band, "INSTR": "WIRCam", "type": "scalar"},
                fieldnames=fieldnames,
                excludeFields=["M31-44","M31-45","M31-46"])

def make_planar_mosaics(bands, includeFields=None):
    """Make WIRCam scalar mosaics, currently excluding the short exposure fields."""
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mosaicDir = "skyoffsets/planar_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    for band in bands:
        mosaicDB = PlanarMosaicFactory(dbname="m31", cname="mosaics")
        mosaicname = "planar_%s"%band
        mosaicDB.make(blockDB, band, mosaicDir, mosaicname,
                solverDBName="skyoffsets",
                nRuns=1,
                resetCouplings=True,
                mosaicMeta={"FILTER": band, "INSTR": "WIRCam", "type": "planar"},
                fieldnames=includeFields,
                excludeFields=["M31-44","M31-45","M31-46"])

def make_alt_planar_mosaics(bands, includeFields=None, nThreads=8, reset=True):
    """Make WIRCam scalar mosaics, currently excluding the short exposure fields."""
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mosaicDir = "skyoffsets/planar_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    for band in bands:
        scalarMosaicDB = MosaicDB(dbname="m31", cname="mosaics")
        doc = scalarMosaicDB.get_mosaic_doc("scalar_%s"%band)
        initOffsets = doc['offsets']

        mosaicDB = AltPlanarMosaicFactory(dbname="m31", cname="mosaics")
        mosaicname = "altplanar_%s"%band
        mosaicDB.init_scalar_offsets(initOffsets)
        mosaicDB.set_plane_distributions(5e-9, 1e-14, 0.5)
        mosaicDB.nThreads = nThreads
        mosaicDB.make(blockDB, band, mosaicDir, mosaicname,
                solverDBName="skyoffsets",
                nRuns=100,
                resetCouplings=reset, # Changed!
                freshStart=reset, #reset,
                mosaicMeta={"FILTER": band, "INSTR": "WIRCam", "type": "planar"},
                fieldnames=includeFields,
                excludeFields=["M31-44","M31-45","M31-46"])

def plot_mosaic(fields):
    """docstring for plot_mosaic"""
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mosaicDir = "skyoffsets/planar_mosaic"

    mosaicDB = AltPlanarMosaicFactory(dbname="m31", cname="mosaics")
    mosaicname = "altplanar_%s" % "J"

    mosaicDB.make_mosaic(blockDB, mosaicname, "J", mosaicDir,
            fieldnames=fields,
            excludeFields=["M31-44","M31-45","M31-46"])
    mosaicDB.subsample_mosaic(mosaicname, pixelScale=1.)

def midgalaxy_alt_plane_mosaic_experiment(nThreads=4, nTrials=1000, reset=False):
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mosaicDir = "skyoffsets/planar_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    band = "J"
    mosaicname = "midgalaxy_altplanar_J"

    fields = ["M31-5","M31-9","M31-13","M31-17","M31-36",
            "M31-6","M31-10","M31-35","M31-34","M31-37","M31-14","M31-18",
            "M31-22","M31-11","M31-15","M31-19","M31-23","M31-39"]

    scalarMosaicDB = MosaicDB(dbname="m31", cname="mosaics")
    doc = scalarMosaicDB.get_mosaic_doc("scalar_%s"%band)
    initOffsets = doc['offsets']

    mosaicDB = AltPlanarMosaicFactory(dbname="m31", cname="mosaics")
    mosaicDB.init_scalar_offsets(initOffsets)
    mosaicDB.set_plane_distributions(5e-9, 5e-13, 0.1)
    mosaicDB.nThreads = nThreads
    mosaicDB.make(blockDB, band, mosaicDir, mosaicname,
            solverDBName="skyoffsets",
            nRuns=nTrials,
            resetCouplings=reset, # Changed!
            freshStart=reset, #reset,
            mosaicMeta={"FILTER": band, "INSTR": "WIRCam", "type": "midgalaxy_alt"},
            fieldnames=fields,
            excludeFields=["M31-44","M31-45","M31-46"])



if __name__ == '__main__':
    main()
