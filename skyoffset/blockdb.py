import pymongo
import astropy.io.fits


class BlockDB(object):
    """Database interface for blocks: sets of detector stacks within a field.
    """
    def __init__(self, footprintDB, dbname="m31", cname="wircam_blocks",
            url="localhost", port=27017):
        super(BlockDB, self).__init__()
        self.dbname = dbname
        self.cname = cname
        self.url = url
        self.port = port
        self.footprintDB = footprintDB

        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]

    def insert(self, doc):
        """Insert a block"""
        self.collection.save(doc, safe=True)
        if 'image_path' in doc:
            self._add_footprint(doc['image_path'], doc['OBJECT'],
                    doc['FILTER'], instrument=doc['instrument'])

    def update(self, blockName, key, value):
        """docstring for update"""
        self.collection.update({"_id": blockName}, {"$set": {key: value}})
        if 'image_path' == key:
            doc = self.collection.find_one({"_id": blockName})
            print "update", doc
            self._add_footprint(doc['image_path'], doc['OBJECT'],
                    doc['FILTER'], instrument=doc['instrument'])

    def _add_footprint(self, path, fieldname, band, instrument):
        """Add this block to the FootprintDB"""
        header = astropy.io.fits.getheader(path)
        self.footprintDB.new_from_header(header, field=fieldname,
                FILTER=band, kind='block', instrument=instrument)

    def find_one(self, sel):
        """docstring for find_one"""
        return self.collection.find_one(sel)
    
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
