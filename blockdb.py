import pymongo
import os
import shutil
import multiprocessing

import pyfits
import owl.astromatic

from difftools import Couplings
from multisimplex import SimplexScalarOffsetSolver
import offsettools


class BlockDB(object):
    """Database interface for blocks: sets of detector stacks within a field."""
    def __init__(self, dbname="m31", cname="wircam_blocks", url="localhost", port=27017):
        super(BlockDB, self).__init__()
        self.dbname = dbname
        self.cname = cname
        self.url = url
        self.port = port

        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]
    
    def find_blocks(self, sel):
        """Find blocks given a MongoDB selector.
        
        e.g. sel={"FILTER":"J"}
        
        :return: a dictionary of block documents.
        """
        docs = {}
        recs = self.collection.find(sel)
        for rec in recs:
            blockName = rec['_id']
            docs[blockName] = rec
        return docs

    def add_all_footprints(self, footprintDB, instrument="WIRCam"):
        """Persist all blocks to the FootprintDB."""
        recs = self.collection.find()
        for rec in recs:
            path = rec['image_path']
            print path
            header = pyfits.getheader(path)
            footprintDB.new_from_header(header, field=rec['field'],
                    FILTER=rec['FILTER'], kind="block",
                    instrument=instrument)
    
    def make(self, stackDB, fieldname, band, workDir, solverCName,
            freshStart=True):
        """Make a block for the given field/band combination.
        :param stackdB: a StackDB instance
        :param fieldname: the field of this block
        :param band: the FILTER of stacks
        :param workDir: directory where blocks are made, the block for this
            field is actually made in workDir/fieldname_band
        :param solverCName: name of the MongoDB collection to store the
            real-time optimization results in from individual runs.
        :param freshStart: True, will cause the solver's mongodb collection
            to be reset. Set to False to build upon results of previous
            solver runs.
        """
        blockName = "%s_%s" % (fieldname, band)
        self.workDir = os.path.join(workDir, blockName)
        if os.path.exists(self.workDir) is False: os.makedirs(self.workDir)
        stackDocs = stackDB.find_stacks({"OBJECT":fieldname,"FILTER":band})
        # Make couplings
        couplings = self._make_couplings(blockName, stackDocs)
        # Add this block to the BlockDB collection
        doc = {"_id": blockName, "field": fieldname, "FILTER":band,
            "solver_cname": solverCName}
        self._init_doc(doc)
        # Perform optimization
        solver = self._solve_offsets(blockName, stackDocs, couplings, solverCName,
                freshStart=freshStart)
        offsets = solver.find_best_offsets() # offset dictionary
        self.collection.update({"_id":blockName}, {"$set":{"offsets": offsets}})
        self.make_block_mosaic(stackDB, fieldname, band, workDir)
    
    def _make_couplings(self, blockName, stackDocs):
        """Computes the couplings between stackDocs.
        :return: a difftools.Couplings instance.
        """
        couplings = Couplings()
        for stackName, stackDoc in stackDocs.iteritems():
            stackPath = stackDoc['image_path']
            stackWeightPath = stackDoc['weight_path']
            couplings.add_field(stackName, stackPath, stackWeightPath)
        diffImageDir = os.path.join(self.workDir, "diffs")
        couplings.make(diffImageDir)
        # pairNames, diffs, diffSigmas, allFields = couplings.get_field_diffs()
        shutil.rmtree(diffImageDir)
        return couplings
    
    def _solve_offsets(self, blockName, stackDocs, couplings, solverCName,
            freshStart=True):
        """Use SimplexScalarOffsetSolver to derive offsets for this block."""
        logPath = os.path.join(self.workDir, "%s.log"%blockName)
        solver = SimplexScalarOffsetSolver(dbname=self.dbname,
                cname=solverCName, url=self.url, port=self.port)
        if freshStart:
            solver.resetdb()
        solver.multi_start(couplings, 10000, logPath, mp=False, cython=True)
        return solver
    
    def _init_doc(self, doc):
        """Insert a new document for this block, deleting an older document
        if necessary.
        """
        blockName = doc['_id']
        if self.collection.find({"_id":blockName}).count() > 0:
            self.collection.remove({"_id":blockName})
        self.collection.insert(doc)

    def make_block_mosaic(self, stackDB, fieldname, band, workDir):
        """The block mosaic can be made anytime once entries are added
        to the solver's collection."""
        blockName = "%s_%s" % (fieldname, band)
        self.workDir = os.path.join(workDir, blockName)
        blockDoc = self.collection.find_one({"_id":blockName})
        solverCName = blockDoc['solver_cname']
        stackDocs = stackDB.find_stacks({"OBJECT":fieldname,"FILTER":band})
        solver = SimplexScalarOffsetSolver(dbname=self.dbname,
                cname=solverCName, url=self.url, port=self.port)
        offsets = solver.find_best_offsets()

        self.offsetDir = os.path.join(self.workDir, "offset_images")
        if os.path.exists(self.offsetDir) is False: os.makedirs(self.offsetDir)

        # apply offsets to images, then swarp
        args = []
        offsetPaths = {}
        weightPaths = {}
        fieldNames = []
        for stackName, stackDoc in stackDocs.iteritems():
            print "stackDoc:", stackDoc
            stackName = stackDoc['_id']
            fieldNames.append(stackName)
            imagePath = stackDoc['image_path']
            weightPaths[stackName] = stackDoc['weight_path']
            offset = offsets[stackName]
            offsetPath = os.path.join(self.offsetDir, stackName+".fits")
            offsetPaths[stackName] = offsetPath
            arg = (stackName, imagePath, offset, offsetPath)
            args.append(arg)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pool.map(offsettools.apply_offset, args)
        
        imagePathList= [offsetPaths[k] for k in fieldNames]
        weightPathList = [weightPaths[k] for k in fieldNames]

        # By not resampling, it means that the block will still be in the
        # same resampling system as all other blocks (hopefully)
        swarpConfigs = {"WEIGHT_TYPE": "MAP_WEIGHT",
                "COMBINE_TYPE": "WEIGHTED",
                "RESAMPLE":"N", "COMBINE":"Y"}
        swarp = owl.astromatic.Swarp(imagePathList, blockName,
                weightPaths=weightPathList, configs=swarpConfigs,
                workDir=self.workDir)
        swarp.run()
        blockPath, weightPath = swarp.getMosaicPaths()
        self.collection.update({"_id":blockName},
                {"$set":{"image_path":blockPath,"weight_path":weightPath}})

        # Delete offset images
        shutil.rmtree(self.offsetDir)

if __name__ == '__main__':
    from andpipe.footprintdb import FootprintDB
    footprintDB = FootprintDB()
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    blockDB.add_all_footprints(footprintDB)
