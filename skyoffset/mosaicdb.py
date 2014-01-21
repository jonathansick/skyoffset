import os
import pymongo

from andpipe import footprintdb
from difftools import CoupledPlanes  # Planar couplings
from planarmultisimplex import PlanarMultiStartSimplex
from planaraltmultisimplex import AltPlanarMultiStartSimplex
import offsettools
import blockmosaic  # common mosaic construction functions


class MosaicDB(object):
    """Database interface for a mosaic: a set of blocks."""
    def __init__(self, dbname="m31", cname="mosaics", url="localhost",
            port=27017):
        super(MosaicDB, self).__init__()
        self.dbname = dbname
        self.cname = cname
        self.url = url
        self.port = port
        
        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]
    
    def find_mosaics(self, sel):
        """Find mosaics given a MongoDB selector."""
        docs = {}
        recs = self.collection.find(sel)
        for rec in recs:
            mosaicName = rec['_id']
            docs[mosaicName] = rec
        return recs

    def get_mosaic_doc(self, mosaicName):
        """Return document for a single, named, mosaic."""
        mosaicDoc = self.collection.find_one({"_id": mosaicName})
        return mosaicDoc

    def insert(self, mosaicDoc):
        """docstring for insert"""
        self.collection.save(mosaicDoc)

    def find_one(self, sel):
        """docstring for find_one"""
        if type(sel) is not dict:
            sel = {"_id": sel}
        return self.collection.find_one(sel)

    def update(self, sel, doc):
        """docstring for update"""
        self.collection.update(sel, doc)


class PlanarMosaicFactory(MosaicDB):
    """Specialized version of MosaicDB for creating planar-offset Mosaics."""
    def __init__(self, *args, **kwargs):
        super(PlanarMosaicFactory, self).__init__(*args, **kwargs)
        self.nThreadsCouplings = 8 # number of threads to use for couplings comp
        

    def _reload_couplings(self, couplingsDocument):
        """Attempt to create a CoupledPlanes instance from a MongoDB
        persisted document."""
        return CoupledPlanes.load_doc(couplingsDocument)
    
    def _make_couplings(self, mosaicName, blockDocs):
        """Computes the couplings between blockDocs.
        :return: a difftools.Couplings instance.
        """
        couplings = CoupledPlanes()
        for blockName, blockDoc in blockDocs.iteritems():
            blockPath = blockDoc['image_path']
            blockWeightPath = blockDoc['weight_path']
            couplings.add_field(blockName, blockPath, blockWeightPath)
        diffImageDir = os.path.join(self.workDir, "diffs")
        couplings.make(diffImageDir, nThreads=self.nThreadsCouplings)
        couplingsDoc = couplings.get_doc()
        print couplingsDoc
        self.collection.update({"_id": mosaicName},
                {"$set": {"couplings": couplingsDoc}})
        return couplings
    
    def _solve_offsets(self, mosaicName, solverDBName, couplings,
            blockWCSs, mosaicResampledWCS, freshStart=True, nRuns=1000):
        """Use SimplexScalarOffsetSolver to derive offsets for this block."""
        logPath = os.path.join(self.workDir, "%s.log"%mosaicName)
        solver = PlanarMultiStartSimplex(dbname=solverDBName, cname=mosaicName,
                url=self.url, port=self.port)

        if freshStart:
            solver.resetdb()
        solver.multi_start(couplings, nRuns, logPath, cython=True, mp=True)
        
        offsets = solver.find_best_offsets(blockWCSs, mosaicResampledWCS)
        self.collection.update({"_id":mosaicName},
                {"$set":{"offsets": offsets,
                    "solver_cname": mosaicName,
                    "solver_db": solverDBName}})
        return solver

    def make_mosaic(self, blockDB, mosaicName, band, workDir, fieldnames=None,
            excludeFields=None, threads=8):
        """Swarp a mosaic using the optimal sky plane offsets.
        
        The mosaic can be made anytime once entries are added
        to the solver's collection. This is because we initialize
        a SimplexScalarOffsetSolver that re-generates the list of best
        offsets from the collection of solver documents.
        """
        self.workDir = os.path.join(workDir, mosaicName)
        mosaicDoc = self.collection.find_one({"_id":mosaicName})
        solverCName = mosaicDoc['solver_cname']
        solverDBName = mosaicDoc['solver_db']

        blockSel = {"FILTER":band}
        if fieldnames is not None or excludeFields is not None:
            blockSel["field"] = {}
        if fieldnames is not None:
            blockSel["field"].update({"$in": fieldnames})
        if excludeFields is not None:
            blockSel["field"].update({"$nin": excludeFields})
        blockDocs = blockDB.find_blocks(blockSel)
        solver = PlanarMultiStartSimplex(dbname=solverDBName, cname=solverCName,
                url=self.url, port=self.port)
        
        footprintSelector = {"mosaic_name": mosaicName,
            "FILTER": mosaicDoc['FILTER'],
            "kind": "mosaic", 'instrument': "WIRCam"}
        footprintDB = footprintdb.FootprintDB()
        mosaicWCS = footprintDB.make_resampled_wcs(footprintSelector)
        blockWCSs = {}
        for blockName, blockDoc in blockDocs.iteritems():
            print "the blockDoc:", blockDoc
            field = blockDoc['field']
            band = blockDoc['FILTER']
            sel = {"field": field, "FILTER": band}
            #blockName = "%s_%s" % (field, band) # changed
            print blockName
            blockWCSs[blockName] = footprintDB.make_resampled_wcs(sel)

        offsets = solver.find_best_offsets(blockWCSs, mosaicWCS)
        print "Using offsets", offsets
        
        blockPath, weightPath = blockmosaic.block_mosaic(blockDocs, offsets,
                mosaicName, self.workDir,
                offset_fcn=offsettools.apply_planar_offset,
                threads=threads)

        self.collection.update({"_id": mosaicName},
                {"$set":{"image_path":blockPath,"weight_path": weightPath}})

class AltPlanarMosaicFactory(PlanarMosaicFactory):
    """Makes planar-optimizes mosaics using the alternating planes method."""
    def __init__(self, **kwargs):
        super(AltPlanarMosaicFactory, self).__init__(**kwargs)
        self.nThreads = 8
    
    def init_scalar_offsets(self, offsets):
        """Field dictionary of initial best-estimates of levels (say from
        the scalar offset solver
        """
        self.initLevels = offsets

    def set_plane_distributions(self, levelSigma, slopeSigma, restartFraction):
        """Makes setting the guassian widths of random levels and slopes more
        accessible."""
        self.levelSigma = levelSigma
        self.slopeSigma = slopeSigma
        self.restartFraction = restartFraction

    def _solve_offsets(self, mosaicName, solverDBName, couplings,
            blockWCSs, mosaicResampledWCS, freshStart=True, nRuns=1000,
            levelSigma=5e-9, slopeSigma=5e-13, restartFraction=0.1):
        """Use AltPlanarMultiStartSimplex to find offset planes for the mosaic."""
        logPath = os.path.join(self.workDir, "%s.log"%mosaicName)
        solver = AltPlanarMultiStartSimplex(dbname=solverDBName, cname=mosaicName,
                url=self.url, port=self.port)

        if freshStart:
            solver.resetdb()

        solver.multi_start(couplings, self.initLevels, nRuns, logPath,
                levelSigma=self.levelSigma, slopeSigma=self.slopeSigma,
                restartFraction=self.restartFraction, mp=True, nThreads=self.nThreads)
        
        offsets = solver.find_best_offsets(blockWCSs, mosaicResampledWCS)
        self.collection.update({"_id":mosaicName},
                {"$set":{"offsets": offsets,
                    "solver_cname": mosaicName,
                    "solver_db": solverDBName}})
        return solver

