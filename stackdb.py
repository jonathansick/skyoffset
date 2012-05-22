import pymongo


class StackDB(object):
    """Database interface for detector field stacks"""
    def __init__(self, dbname="m31", cname="mosaic_stacks", url="localhost",
            port=27017):
        super(StackDB, self).__init__()
        self.dbname = dbname
        self.cname = cname
        self.url = url
        self.port = port
        
        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]

    def insert(self, doc):
        """docstring for insert"""
        self.collection.insert(doc)

    def find_stacks(self, sel):
        """Does a MongoDB query for stacks, returning a dictionary (keyed) by
        stack name of the stack documents.
        :param sel: a MongoDB query dictionary.
        """
        recs = self.collection.find(sel)
        docDict = {}
        for r in recs:
            stackName = r['_id']
            docDict[stackName] = r
        return docDict
