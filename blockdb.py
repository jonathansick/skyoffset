import pymongo
import pyfits

from andpipe.footprintdb import FootprintDB


class BlockDB(object):
    """Database interface for blocks: sets of detector stacks within a field.
    """
    def __init__(self, dbname="m31", cname="wircam_blocks", url="localhost",
            port=27017):
        super(BlockDB, self).__init__()
        self.dbname = dbname
        self.cname = cname
        self.url = url
        self.port = port
        self.footprintDB = FootprintDB()

        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]

    def insert(self, doc):
        """Insert a block"""
        self.collection.insert(doc)
        self._add_footprint(doc['image_path'], doc['field'], doc['FILTER'],
                instrument=doc['instrument'])

    def update(self, blockName, key, value):
        """docstring for update"""
        self.collection.update({"_id": blockName}, {"$set": {key: value}})

    def _add_footprint(self, path, fieldname, band, instrument):
        """Add this block to the FootprintDB"""
        header = pyfits.getheader(path)
        self.footprintDB.new_from_header(header, field=fieldname,
                FILTER=band, kind='block', instrument=instrument)
    
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
