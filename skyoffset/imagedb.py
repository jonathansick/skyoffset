#!/usr/bin/env python
# encoding: utf-8
"""
MongoDB databases for storing references to mosaic image products (at the
stack, block and mosaic level).

The databases are backed by the Mo'Astro ImageLog API. Most of the additional
methods provided for each database are to preserve backwards compatibility.
"""

import astropy.io.fits
from moastro.imagelog import ImageLog


class StackDB(ImageLog):
    """Database interface for detector field stacks."""
    def __init__(self, dbname, cname,
            server=None, url="localhost", port=27017):
        super(StackDB, self).__init__(dbname, cname,
                server=server, url=url, port=port)

    def find_mosaics(self, sel):
        """Find mosaics given a MongoDB selector."""
        docs = {}
        recs = self.c.find(sel)
        for rec in recs:
            mosaicName = rec['_id']
            docs[mosaicName] = rec
        return recs

    def get_mosaic_doc(self, mosaicName):
        """Return document for a single, named, mosaic."""
        mosaicDoc = self.c.find_one({"_id": mosaicName})
        return mosaicDoc

    def insert(self, mosaicDoc):
        """Insert a stack document into the collection."""
        self.c.save(mosaicDoc)

    def find_one(self, sel):
        """Return the document for a single stack."""
        if type(sel) is not dict:
            sel = {"_id": sel}
        return self.c.find_one(sel)

    def update(self, sel, doc):
        """Update the selected stack document."""
        self.c.update(sel, doc)


class BlockDB(ImageLog):
    """Database interface for blocks: sets of detector stacks within a field.
    """
    def __init__(self, footprintDB, dbname, cname,
            server=None, url="localhost", port=27017):
        super(BlockDB, self).__init__(dbname, cname,
                server=server, url=url, port=port)
        self.footprintDB = footprintDB

    def insert(self, doc):
        """Insert a block"""
        self.c.save(doc, safe=True)
        if 'image_path' in doc:
            self._add_footprint(doc['image_path'], doc['OBJECT'],
                    doc['FILTER'], instrument=doc['instrument'])

    def update(self, blockName, key, value):
        """docstring for update"""
        self.c.update({"_id": blockName}, {"$set": {key: value}})
        if 'image_path' == key:
            doc = self.c.find_one({"_id": blockName})
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
        return self.c.find_one(sel)
    
    def find_blocks(self, sel):
        """Find blocks given a MongoDB selector.
        
        e.g. sel={"FILTER":"J"}
        
        :return: a dictionary of block documents.
        """
        docs = {}
        recs = self.c.find(sel)
        for rec in recs:
            blockName = rec['_id']
            docs[blockName] = rec
        return docs


class MosaicDB(ImageLog):
    """Database interface for a mosaic: a set of blocks."""
    def __init__(self, dbname, cname,
            server=None, url="localhost", port=27017):
        super(MosaicDB, self).__init__(dbname, cname,
                server=server, url=url, port=port)
    
    def find_mosaics(self, sel):
        """Find mosaics given a MongoDB selector."""
        docs = {}
        recs = self.c.find(sel)
        for rec in recs:
            mosaicName = rec['_id']
            docs[mosaicName] = rec
        return recs

    def get_mosaic_doc(self, mosaicName):
        """Return document for a single, named, mosaic."""
        mosaicDoc = self.c.find_one({"_id": mosaicName})
        return mosaicDoc

    def insert(self, mosaicDoc):
        """docstring for insert"""
        self.c.save(mosaicDoc)

    def find_one(self, sel):
        """docstring for find_one"""
        if type(sel) is not dict:
            sel = {"_id": sel}
        return self.c.find_one(sel)

    def update(self, sel, doc):
        """docstring for update"""
        self.c.update(sel, doc)
