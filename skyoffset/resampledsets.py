import pymongo
import os
import shutil
import pyfits

from andpipe.imagelog import WIRLog
#from resampling import CalibratedResampling

def main():
    print "bootstraping"
    bootstrap_wircam_resampled_dataset()
    #clean_resampledsets()

def bootstrap_wircam_resampled_dataset():
    """Fills in the Resampled Dataset using the results of wirpipe. Does not
    resampling through ResampledDataset itself!"""
    imageLog = WIRLog()
    resampledDataset = ResampledDataset(imageLog, dbname="m31",
            cname="wircam_resampled", url="localhost", port=27017)
    resampledDataset.exts = [1,2,3,4]
    imageKeys = imageLog.search({"TYPE":'sci'})
    print "Found %i images" % len(imageKeys)
    #resampledDataset.imageKeys = imageKeys
    #resampledDataset._create_docs() # initialize docs for these image keys

    exts = [1,2,3,4]
    #wirpipeDir = "/Volumes/Zaphod/m31/wirpipe/skyoffsets_window_zp/resampledimages/resample"
    destDir = "skyoffsets/resampledset"
    wirpipeDir = "skyoffsets/resampledset"
    if os.path.exists(destDir) is False: os.makedirs(destDir)
    resampledImagePaths = {}
    resampledWeightPaths = {}
    docs = []
    allFrames = []
    for imageKey in imageKeys:
        for ext in exts:
            frame = "%s_%i"%(imageKey,ext)
            #wirpipeImage = os.path.join(wirpipeDir, "%s_%i.resamp.fits"%(imageKey,ext))
            #wirpipeWeight = os.path.join(wirpipeDir, "%s_%i.resamp.weight.fits"%(imageKey,ext))
            wirpipeImage = os.path.join(destDir, "%s_%i.fits"%(imageKey,ext))
            wirpipeWeight = os.path.join(destDir, "%s_%i_weight.fits"%(imageKey,ext))
            if os.path.exists(wirpipeImage) and os.path.exists(wirpipeWeight):
                destImage = os.path.join(destDir, "%s_%i.fits"%(imageKey,ext))
                destWeight = os.path.join(destDir, "%s_%i_weight.fits"%(imageKey,ext))
                #shutil.copy(wirpipeImage, destImage)
                #shutil.copy(wirpipeWeight, destWeight)
                # Add image to ResampledDataSet
                resampledImagePaths[frame] = destImage
                resampledWeightPaths[frame] = destWeight
                allFrames.append(frame)
                doc = {"_id":frame, "resampled_path": destImage,
                        "resampled_weight_path": destWeight}
                docs.append(doc)
            else:
                print "cant find", frame
    print "There are %i images"%len(resampledImagePaths.keys())
    resampledDataset.collection.drop()
    resampledDataset.collection = resampledDataset.db["wircam_resampled"]
    resampledDataset.collection.insert(docs)
    #resampledDataset.imageKeys = list(set(resampledImagePaths.keys()))
    #resampledDataset._create_docs() # initialize docs for these image keys
    #resampledDataset._save_path_data("resampled_path", resampledImagePaths)
    #resampledDataset._save_path_data("resampled_weight_path", resampledWeightPaths)

def clean_resampledsets():
    """docstring for clean_resampledsets"""
    imageLog = WIRLog()
    resampledDataset = ResampledDataset(imageLog, dbname="m31",
            cname="wircam_resampled", url="localhost", port=27017)
    docs = resampledDataset.collection.find({})
    for doc in docs:
        name = doc['_id']
        if 'resampled_path' not in doc:
            print "removing", name
            resampledDataset.collection.remove({"_id":name})


class ResampledDataset(object):
    """Coordinates a group of frames that are resampled for mosaicing and
    offset analysis.

    The mongodb collection is structured by frame (named by imageKey_ext):

    * imageKey_ext
        - image_key : imageKey
        - ext : ext
        - weight_path : weightPath (as created by prep_weight_maps)
        - source_path : sourcePath (as installed from the MEF)
        - resampled_path : path to resampled image
        - resampled_weight_path : path to resampled weight image
    """
    def __init__(self, imageLog, dbname="m31", cname="wircam_resampled",
            url="localhost", port=27017):
        super(ResampledDataset, self).__init__()
        self.imageLog = imageLog
        self.dbname = dbname
        self.cname = cname
        self.url= url
        self.port = port
        
        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]

        
    def _make_frame_list(self):
        """Creates a list of frame keywords."""
        frames = []
        for imageKey in self.imageKeys:
            for ext in self.exts:
                frames.append("%s_%i" % (imageKey, ext))
        return frames

    def _create_docs(self):
        """Ensures that a document exists for each frame, creating one if
        necessary.
        """
        frameNames = self._make_frame_list()
        records = self.collection.find({"_id":{"$in":frameNames}}, {"_id":1})
        existingFrameNames = set([rec['_id'] for rec in records])
        frameNames = set(frameNames)
        # Get frame names of frames not already in documents, but that are
        # in frameNames
        unCreatedFrames = list(frameNames - existingFrameNames)
        if len(unCreatedFrames) > 0:
            # These unCreatedFrames are what we need to initialize
            newDocs = [{"_id":f} for f in unCreatedFrames]
            self.collection.insert(newDocs)
    
    def _get_path_dictionary(self, frames, key):
        """Makes a dictionary of paths loaded from the MongoDB collection."""
        dataDict = {}
        recs = self.collection.find({"_id":{"$in":frames}})
        for r in recs:
            print r
            frame = r['_id']
            data = r[key]
            dataDict[frame] = data
        return dataDict
    
    def _save_path_data(self, key, pathDict):
        """Saves a dictionary of path data back to MongoDB."""
        for frame, path in pathDict.iteritems():
            self.collection.update({"_id":frame}, {"$set": {key: path}})

    def resample_images(self, imageKeys, makeWeights=True, install=True,
            resample=True, exts=[1,2,3,4]):
        """Resamples the dataset. First installs the images, then resamples,
        then saves the resampled image paths to the image log.
        """
        self.imageKeys = imageKeys
        self.exts = exts
        frameNames = self._make_frame_list()
        self._create_docs()
        
        resampler = CalibratedResampling(self.imageLog, imageKeys,
                        self.resamplingWorkDir, exts=self.exts)
        
        if makeWeights:
            weightPaths = resampler.prep_weight_maps(self.imageSourceKey)
            self._save_path_data("weight_path", weightPaths)
        else:
            weightPaths = self._get_path_dictionary(frameNames, 'weight_path')
            resampler.set_weight_maps(weightPaths)
        
        if install:
            imagePaths = resampler.installForResampling(imageKeys, [1,2,3,4],
                    self.imageSourceKey, self.zeropointKey)
            self._save_path_data("source_path", imagePaths)
        else:
            imagePaths = self._get_path_dictionary(frameNames, 'source_path')
            resampler.set_source_paths(imagePaths)

        if resample:
            resampledImagePaths, resampledWeightPaths = resampler.resample()
            self._save_path_data("resampled_path", resampledImagePaths)
            self._save_path_data("resampled_weight_path", resampledWeightPaths)

    def get_paths(self, imageKeys, ext):
        """:return: dicts of *resampled* image paths and weight paths"""
        frames = ["%s_%i"%(ik,ext) for ik in imageKeys]
        imagePathDict = self._get_path_dictionary(frames, 'resampled_path')
        weightPathDict = self._get_path_dictionary(frames, 'resampled_weight_path')
        #imagePaths = [imagePathDict[f] for f in frames]
        #weightPaths = [weightPathDict[f] for f in frames]
        return imagePathDict, weightPathDict

class CalibratedResampling(object):
    """Resamples frames, and handles zeropoints."""
    def __init__(self, imageLog, imageKeys, workDir, exts=[1,2,3,4]):
        super(CalibratedResampling, self).__init__()
        self.imageLog = imageLog
        self.imageKeys = imageKeys
        self.workDir = workDir
        self.exts = exts # list of extensions to act upon

        self.sourcePaths = {}
        self.weightPaths = {}

        self.name = "m31_resampled"
        
    def prep_weight_maps(self, pathKey):
        """Make weight maps for the images.
        TODO right now I'm re-using the weightmaps constructed by wirpipe
        This gets around issues of flagmaps.
        :return: dictionary of weight paths, keyed by frame name.
        """
        weightPaths = {}
        for imageKey in self.imageKeys:
            for ext in self.exts:
                frame = "%s_%i"%(imageKey, ext)
                weightPath = os.path.join(self.workDir, "sources")
                weightPaths[frame] = weightPath
        return weightPaths
    
    def set_weight_maps(self, weightPaths):
        """Set the weight paths dictionary (rather than create new ones)."""
        self.weightPaths = weightPaths

    def install(self, pathKey, zpKey):
        """Install images while applying their zeropoint calibrations.
        TODO take care of weight maps.
        """
        self.installDir = os.path.join(self.workDir, 'sources')
        if os.path.exists(self.installDir) is False:
            os.makedirs(self.installDir)
        
        for imageKey in self.imageKeys:
            for ext in self.exts:
                frame = (imageKey, ext)
                # TODO should do batch mongodb queries instead
                imagePath = self.imageLog[imageKey][pathKey]
                zp = self.imageLog[imageKey]["%i.%s"%(ext,zpKey)]
                exptime = self.imageLog[imageKey]['EXPTIME']
                fits = pyfits.open(imagePath)
                image = fits[ext].data
                header = fits[ext].header
                fits.close()
                
                fscale = 10.**(-2.*zp/5.) / exptime
                image = image * fscale

                installPath = os.path.join(self.installDir,
                        "%s_%i.fits"%(imageKey, ext))
                header.update("FLXSCALE", 1.0) # prevent Swarp flux scaling

                pyfits.writeto(installPath, image, header, clobber=True)
                self.sourcePaths[frame] = installPath
    
    def set_source_paths(self, sourcePaths):
        """Set source image paths if they don't need to be installed."""
        self.sourcePaths = sourcePaths

    def resample(self, pixscale=None):
        """Resample images to a common system."""
        mosaicPath = os.path.join(self.workDir, self.name)
        self.resampleDir = os.path.join(self.workDir, "resample")
        if os.path.exists(self.resampleDir) is False: os.makedirs(self.resampleDir)
        frames = self.sourcePaths.keys()
        imagePathList = [self.sourcePaths[frame] for frame in frames]
        weightPathList = [self.weightPaths[frame] for frame in frames]
        configs = {"RESAMPLE_DIR": self.resampleDir,
            "COMBINE_TYPE":"AVERAGE", "WEIGHT_TYPE":"MAP_WEIGHT",
            "NTHREADS":"8", "SUBTRACT_BACK":"N",
            "WRITE_XML":"N", "VMEM_DIR":"/Volumes/Zaphod/tmp",
            "MEM_MAX": "8000"}
        if self.pixscale is not None:
            configs['PIXEL_SCALE'] = "%.2f" % self.pixscale
            configs['PIXELSCALE_TYPE'] = 'MANUAL'
        # TODO change to the new Swarp
        swarp = owl.Swarp.Swarp2(imagePathList, self.name+".fits",
            weightPaths=weightPathList, configs=configs, workDir=self.workDir)
        swarp.run()

        return self._make_resampled_paths
    
    def _make_resampled_paths(self, frames, resampleDir):
        """docstring for _makeResampledPaths"""
        resampledPaths = {}
        resampledWeightPaths = {}
        for imageKey in self.imageKeys:
            for ext in self.exts:
                frame = "%s_%i"%(imageKey, ext)
                sourcePath = self.sourcePaths[frame]
                sourceRoot = os.path.splitext(os.path.basename(sourcePath))[0]
                resamplePath = os.path.join(resampleDir, sourceRoot+".resamp.fits")
                resampledWeightPath = os.path.join(resampleDir, sourceRoot+".resamp.weight.fits")
                resampledPaths[frame] = resamplePath
                resampledWeightPaths[frame] = resampledWeightPath
        return resampledPaths, resampledWeightPaths

if __name__ == '__main__':
    main()
