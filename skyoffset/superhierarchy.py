#!/usr/bin/env python
# encoding: utf-8
"""
Make mosaics with super hierarchy: create mosaics from small sets fo blocks,
then put those blocks together in the optimization.

History
-------
2011-09-19 - Created by Jonathan Sick

"""

__all__ = ['']

import os
from blockdb import BlockDB
from mosaicdb import MosaicDB
from mosaicdb import AltPlanarMosaicFactory

nameit = lambda lst: ["M31-%i"%i for i in lst]
nameblocks = lambda lst, band: ["altplanar_cluster_%s_%s"%(i,band) for i in lst]

def main():
    band = "Ks"
    #superblocks = define_super2_blocks() # HIJ1J2KLMNO
    #make_all_clusters(superblocks, band)
    #copy_all_superblocks(superblocks, band)
    #make_alt_planar_supercluster_mosaic(band, "HIJ1J2KLMNO",
    #    superblocks.keys(),
    #     nThreads=7, reset=True)
    make_alt_planar_supercluster_mosaic(band, "IJ1J2KLMNO",
         ["I","J1","J2","K","L","M","N","O"],
         nThreads=7, reset=True)

def make_all_clusters(clusters, band):
    """docstring for make_all_clusters"""
    for clusterName, fields in clusters.iteritems():
        make_alt_planar_cluster_mosaic(band, clusterName, fields,
                nThreads=7, reset=True)

def copy_all_superblocks(clusters, band):
    """docstring for copy_all_superblocks"""
    for clusterName, fields in clusters.iteritems():
        mosaicName = "altplanar_cluster_%s_%s"% (clusterName, band)
        add_to_blockdb(mosaicName)
    # automatically load these block's footprints into FootprintDB
    from andpipe.footprintdb import FootprintDB
    footprintDB = FootprintDB()
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    blockDB.add_all_footprints(footprintDB)


def define_clusters():
    clusters = {"A": nameit([1,3,2,4,32]),
            "B": nameit([7,8,12,33,34]),
            "C": nameit([5,9,6,10,35,37]),
            "D": nameit([13,17,14,18,22,36,38]),
            "E": nameit([11,15,16,40,39,41]),
            "F": nameit([20,21,24,25,26,27,42])}
    return clusters

def define_super_blocks():
    """Super blocks are collections of blocks.
    This series is purposefully designed to have overlaps for a
    better mosaicing of super blocks.
    It also purposefully ignores M31-43 which has a poor coupling
    that needs to be filtered in a code fix"""
    superblocks = {"H": nameit([1,3,2,4,32,7,8]), # extra 7,8
                   "I": nameit([7,8,12,33,9,13]), # extra 9,13
                   "J": nameit([5,9,13,6,10,14,34,35,37]), # extra 13,14
                   "K": nameit([13,17,36,14,18,22,38]),
                   "L": nameit([10,14,11,15,40,39,37]),
                   "M": nameit([19,23,20,24,18,22]),
                   "N": nameit([16,20,21,40,42,24,25]),
                   "O": nameit([20,24,26,21,25,27,41]) # ignore 42!
                   }
    return superblocks

def define_splitJ_superblock():
    superblocks = {"J1": nameit([5,9,6,10,35,34]),
                   "J2": nameit([9,10,13,14,37])}
    return superblocks

def define_super2_blocks():
    """The HIJ1J2KLMNO set
    It also purposefully ignores M31-43 which has a poor coupling
    that needs to be filtered in a code fix"""
    superblocks = {"H": nameit([1,3,2,4,32,7,8]), # extra 7,8
                   "I": nameit([7,8,12,33,9,13]), # extra 9,13
                   "J1": nameit([5,9,6,10,35,34]),
                   "J2": nameit([9,10,13,14,37]),
                   "K": nameit([13,17,36,14,18,22,38]),
                   "L": nameit([10,14,11,15,40,39,37]),
                   "M": nameit([19,23,20,24,18,22]),
                   "N": nameit([16,20,21,40,42,24,25]),
                   "O": nameit([20,24,26,21,25,27,41]) # ignore 42!
                   }
    return superblocks


def make_alt_planar_cluster_mosaic(band, clusterName, includeFields,
        nThreads=8, reset=True):
    """Make WIRCam scalar mosaics, currently excluding the short exposure fields."""
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mosaicDir = "skyoffsets/planar_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    scalarMosaicDB = MosaicDB(dbname="m31", cname="mosaics")
    doc = scalarMosaicDB.get_mosaic_doc("scalar_%s"%band)
    initOffsets = doc['offsets']

    mosaicDB = AltPlanarMosaicFactory(dbname="m31", cname="mosaics")
    mosaicname = "altplanar_cluster_%s_%s"% (clusterName, band)
    mosaicDB.init_scalar_offsets(initOffsets)
    mosaicDB.set_plane_distributions(5e-9, 1e-14, 0.5)
    mosaicDB.nThreads = nThreads
    mosaicDB.make(blockDB, band, mosaicDir, mosaicname,
            solverDBName="skyoffsets",
            nRuns=1000,
            resetCouplings=reset, # Changed!
            freshStart=reset, #reset,
            mosaicMeta={"FILTER": band,
                "INSTR": "WIRCam",
                "type": "planar_cluster",
                "cluster_name": clusterName},
            fieldnames=includeFields,
            excludeFields=["M31-44","M31-45","M31-46"])

def add_to_blockdb(mosaicName):
    """Adds a superblock mosaic into the BlockDB."""
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mosaicDB = MosaicDB(dbname="m31", cname="mosaics")
    mosaicDoc = mosaicDB.get_mosaic_doc(mosaicName)
    mosaicDoc['field'] = mosaicDoc['cluster_name']
    origID = mosaicDoc['_id']
    blockDB.collection.remove({"_id":origID})
    blockDB.collection.insert(mosaicDoc)

def make_alt_planar_supercluster_mosaic(band, mosaicID, includeFields,
        nThreads=8, reset=True):
    """Make a mosaic out of the superblocks. includeFields should be names
    of the super blocks."""
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mosaicDir = "skyoffsets/planar_mosaic"
    if os.path.exists(mosaicDir) is False: os.makedirs(mosaicDir)
    #scalarMosaicDB = MosaicDB(dbname="m31", cname="mosaics")
    #doc = scalarMosaicDB.get_mosaic_doc("scalar_%s"%band)
    #initOffsets = doc['offsets']
    fullBlockNames = nameblocks(includeFields, band)
    initOffsets = {}
    for field in fullBlockNames:
        initOffsets[field] = 0.
    print "init offsets:", initOffsets

    mosaicDB = AltPlanarMosaicFactory(dbname="m31", cname="mosaics")
    mosaicname = "altplanar_superclusters_%s_%s" % (mosaicID, band)
    mosaicDB.nThreadsCouplings = 1 # because these are BIG diff images
    mosaicDB.init_scalar_offsets(initOffsets)
    mosaicDB.set_plane_distributions(5e-9, 1e-14, 0.5)
    mosaicDB.nThreads = nThreads
    mosaicDB.make(blockDB, band, mosaicDir, mosaicname,
            solverDBName="skyoffsets",
            nRuns=1000,
            resetCouplings=reset, # Changed!
            freshStart=reset, #reset,
            mosaicMeta={"FILTER": band,
                "INSTR": "WIRCam",
                "type": "planar_super_cluster"},
            fieldnames=includeFields,
            excludeFields=["M31-44","M31-45","M31-46"])

#assemble_alt_planar_supercluster_mosaic("J",
#            ["A","B","C","D","E","F"])
def assemble_alt_planar_supercluster_mosaic(band, includeFields):
    """docstring for assemble_alt_planar_supercluster_mosaic"""
    mosaicname = "altplanar_superclusters_%s" % band
    mosaicDir = "skyoffsets/planar_mosaic"

    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    
    mosaicDB = AltPlanarMosaicFactory(dbname="m31", cname="mosaics")
    mosaicDB.make_mosaic(blockDB, mosaicname, "J", mosaicDir,
            fieldnames=includeFields,
            excludeFields=["M31-44","M31-45","M31-46"],
            threads=1)
    mosaicDB.subsample_mosaic(mosaicname, pixelScale=1.)


if __name__ == '__main__':
    main()


