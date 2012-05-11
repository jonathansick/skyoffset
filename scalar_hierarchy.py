"""Computes the net sky offset for each frame and tabulates histograms of
offsets at each stage.

Produces the table of hierarchical scalar offsets for skysubpub
"""
import numpy as np
import math
from andpipe.imagelog import WIRLog
from stackdb import StackDB
from blockdb import BlockDB
from mosaicdb import MosaicDB

def main():
    # Writes ext.net_scalar_levels for every frame in imageLog
    #compute_total_scalar_offsets()
    
    # Make table to frame->stack; stack->block; block->mosaic for paper
    print_hierarchy_table()

def compute_total_scalar_offsets():
    """For every frame in the ImageLog, its net offset is computed.
    The result is saved in the image log under ext.net_scalar_offset.
    The net sky level is also computed relative to its associated
    sky image, ext.net_scalar_level (in intensity units)
    """
    imageLog = WIRLog()
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    mosaicDB = MosaicDB()
    stackDB = StackDB(dbname="m31", cname="wircam_stacks")

    for band in ["J","Ks"]:
        mosaicDoc = mosaicDB.get_mosaic_doc("scalar_%s"%band)
        blockOffsets = mosaicDoc['offsets']
        for blockName, blockOffset in blockOffsets.iteritems():
            print blockName, blockOffset
            fieldName = blockName.split("_")[0]
            blockDoc = blockDB.collection.find_one({"FILTER":band,"field":fieldName})
            stackOffsets = blockDoc['offsets']
            for stackName, stackOffset in stackOffsets.iteritems():
                print stackName, stackOffset
                extS = stackName.split("_")[-1]
                stackDoc = stackDB.collection.find_one({"FILTER":band,
                    "OBJECT": fieldName, "ext": int(extS)})
                frameOffsets = stackDoc['offsets']
                for frameKey, frameOffset in frameOffsets.iteritems():
                    print frameKey, frameOffset
                    imageKey = frameKey.split("_")[0]
                    netOffset = frameOffset + stackOffset + blockOffset
                    imageLog.set(imageKey,
                            ".".join((extS,"net_scalar_offset")),netOffset)

def print_hierarchy_table():
    """Makes the table of offset dispersions at each level."""
    frameSigmaJ07_I, frameSigmaJ07_rel = get_frame_sigma("J","2007B")
    frameSigmaK07_I, frameSigmaK07_rel = get_frame_sigma("Ks","2007B")
    frameSigmaJ09_I, frameSigmaJ09_rel = get_frame_sigma("J","2009B")
    frameSigmaK09_I, frameSigmaK09_rel = get_frame_sigma("Ks","2009B")

    stackSigmaJ07_I, stackSigmaJ07_rel = get_stack_sigma("J","2007B")
    stackSigmaK07_I, stackSigmaK07_rel = get_stack_sigma("Ks","2007B")
    stackSigmaJ09_I, stackSigmaJ09_rel = get_stack_sigma("J","2009B")
    stackSigmaK09_I, stackSigmaK09_rel = get_stack_sigma("Ks","2009B")

    blockSigmaJ07_I, blockSigmaJ07_rel = get_block_sigma("J","2007B")
    blockSigmaK07_I, blockSigmaK07_rel = get_block_sigma("Ks","2007B")
    blockSigmaJ09_I, blockSigmaJ09_rel = get_block_sigma("J","2009B")
    blockSigmaK09_I, blockSigmaK09_rel = get_block_sigma("Ks","2009B")

    totalSigmaJ07_I, totalSigmaJ07_rel = get_total_sigma("J","2007B")
    totalSigmaK07_I, totalSigmaK07_rel = get_total_sigma("Ks","2007B")
    totalSigmaJ09_I, totalSigmaJ09_rel = get_total_sigma("J","2009B")
    totalSigmaK09_I, totalSigmaK09_rel = get_total_sigma("Ks","2009B")

    skyJ = 10.**(-2.*19.3/5.) * 0.3*0.3 # J ADU/s/pix^2 from mag/arcsec^2
    skyK = 10.**(-2.*16.6/5.) * 0.3*0.3 # K ADU/s/pix^2 from mag/arcsec^2

    #Thesis values:
    #skyJ = 1.0e-7
    #skyK = 4.3e-7

    mag = lambda I: -2.5*math.log10(I/(0.3*0.3))

    print "type Band, Semester, mag sigma, sigma/sky"
    print "frame J 2007 %.1f %.2f" % (mag(frameSigmaJ07_I), frameSigmaJ07_rel*100.)
    print "frame J 2009 %.1f %.2f" % (mag(frameSigmaJ09_I), frameSigmaJ09_rel*100.)
    print ""
    print "stack J 2007 %.1f %.2f" % (mag(stackSigmaJ07_I), stackSigmaJ07_rel*100.)
    print "stack J 2009 %.1f %.2f" % (mag(stackSigmaJ09_I), stackSigmaJ09_rel*100.)
    print ""
    print "block J 2007 %.1f %.2f" % (mag(blockSigmaJ07_I), blockSigmaJ07_rel*100.)
    print "block J 2009 %.1f %.2f" % (mag(blockSigmaJ09_I), blockSigmaJ09_rel*100.)
    print ""
    print "total J 2007 %.1f %.2f" % (mag(totalSigmaJ07_I), totalSigmaJ07_rel*100.)
    print "total J 2009 %.1f %.2f" % (mag(totalSigmaJ09_I), totalSigmaJ09_rel*100.)
    print "--"
    print "frame K 2007 %.1f %.2f" % (mag(frameSigmaK07_I), frameSigmaK07_rel*100.)
    print "frame K 2009 %.1f %.2f" % (mag(frameSigmaK09_I), frameSigmaK09_rel*100.)
    print ""
    print "stack K 2007 %.1f %.2f" % (mag(stackSigmaK07_I), stackSigmaK07_rel*100.)
    print "stack K 2009 %.1f %.2f" % (mag(stackSigmaK09_I), stackSigmaK09_rel*100.)
    print ""
    print "block K 2007 %.1f %.2f" % (mag(blockSigmaK07_I), blockSigmaK07_rel*100.)
    print "block K 2009 %.1f %.2f" % (mag(blockSigmaK09_I), blockSigmaK09_rel*100.)
    print ""
    print "total K 2007 %.1f %.2f" % (mag(totalSigmaK07_I), totalSigmaK07_rel*100.)
    print "total K 2009 %.1f %.2f" % (mag(totalSigmaK09_I), totalSigmaK09_rel*100.)

    #print "type Band, Semester, mag sigma, sigma/sky"
    #print "frame J 2007 %.1f %.2f" % (mag(frameSigmaJ07_I), frameSigmaJ07_I/skyJ*100.)
    #print "frame J 2009 %.1f %.2f" % (mag(frameSigmaJ09_I), frameSigmaJ09_I/skyJ*100.)
    #print ""
    #print "stack J 2007 %.1f %.2f" % (mag(stackSigmaJ07_I), stackSigmaJ07_I/skyJ*100.)
    #print "stack J 2009 %.1f %.2f" % (mag(stackSigmaJ09_I), stackSigmaJ09_I/skyJ*100.)
    #print ""
    #print "block J 2007 %.1f %.2f" % (mag(blockSigmaJ07_I), blockSigmaJ07_I/skyJ*100.)
    #print "block J 2009 %.1f %.2f" % (mag(blockSigmaJ09_I), blockSigmaJ09_I/skyJ*100.)
    #print ""
    #print "total J 2007 %.1f %.2f" % (mag(totalSigmaJ07_I), totalSigmaJ07_I/skyJ*100.)
    #print "total J 2009 %.1f %.2f" % (mag(totalSigmaJ09_I), totalSigmaJ09_I/skyJ*100.)
    #print "--"
    #print "frame K 2007 %.1f %.2f" % (mag(frameSigmaK07_I), frameSigmaK07_I/skyK*100.)
    #print "frame K 2009 %.1f %.2f" % (mag(frameSigmaK09_I), frameSigmaK09_I/skyK*100.)
    #print ""
    #print "stack K 2007 %.1f %.2f" % (mag(stackSigmaK07_I), stackSigmaK07_I/skyK*100.)
    #print "stack K 2009 %.1f %.2f" % (mag(stackSigmaK09_I), stackSigmaK09_I/skyK*100.)
    #print ""
    #print "block K 2007 %.1f %.2f" % (mag(blockSigmaK07_I), blockSigmaK07_I/skyK*100.)
    #print "block K 2009 %.1f %.2f" % (mag(blockSigmaK09_I), blockSigmaK09_I/skyK*100.)
    #print ""
    #print "total K 2007 %.1f %.2f" % (mag(totalSigmaK07_I), totalSigmaK07_I/skyK*100.)
    #print "total K 2009 %.1f %.2f" % (mag(totalSigmaK09_I), totalSigmaK09_I/skyK*100.)
    #
    #print ""
    #print "Fiducial J sky: %.2f" % mag(skyJ)
    #print "Fiducial Ks sky: %.2f" % mag(skyK)

def get_frame_sigma(band,semester):
    """Compute dispersion of frame offsets for a band/semester."""
    fields = _get_fields(band, semester)
    sel = {"OBJECT": {"$in": fields}, "FILTER": band}
    stackDB = StackDB()
    stackDocs = stackDB.find_stacks(sel)
    offsetLst = []
    relOffsetLst = []
    imageLog = WIRLog()
    for stackName, doc in stackDocs.iteritems():
        #offsetLst += [v for k,v in doc['offsets'].iteritems()]
        for frameKey, offset in doc['offsets'].iteritems():
            imageKey, hdu = frameKey.split("_")
            skyI = paired_sky_intensity(imageLog, imageKey, hdu)
            relOffsetLst.append(offset / skyI)
            offsetLst.append(offset)
    offsets = np.array(offsetLst)
    relOffsets = np.array(relOffsetLst)
    return offsets.std(), relOffsets.std()

def get_stack_sigma(band, semester):
    """Computer dispersion of stack offsets for a band/semester."""
    fields = _get_fields(band, semester)
    sel = {"field": {"$in": fields}, "FILTER": band}
    blockDB = BlockDB(dbname="skyoffsets", cname="wircam_blocks")
    blockDocs = blockDB.find_blocks(sel)
    offsetLst = []
    relOffsetLst = []
    for blockName, doc in blockDocs.iteritems():
        fieldName = blockName.split("_")[0]
        skyI = mean_block_sky_intensity(fieldName, band)
        for stackName, offset in doc['offsets'].iteritems():
            offsetLst.append(offset)
            relOffsetLst.append(offset/skyI)
    offsets = np.array(offsetLst)
    relOffsets = np.array(relOffsetLst)
    return offsets.std(), relOffsets.std()

def get_block_sigma(band, semester):
    """Compute dispersion of block offsets for a band/semester."""
    fields = _get_fields(band, semester)
    mosaicDB = MosaicDB()
    mosaicDoc = mosaicDB.get_mosaic_doc("scalar_%s"%band)
    offsetLst = []
    relOffsetLst = []
    for blockName, blockOffset in mosaicDoc['offsets'].iteritems():
        fieldname = blockName.split("_")[0]
        if fieldname in fields:
            offsetLst.append(blockOffset)
            # Now compute the mean sky level observed while assembling these blocks
            blockSkyIntensity = mean_block_sky_intensity(fieldname, band)
            print blockName, blockOffset, blockSkyIntensity
            relOffsetLst.append(blockOffset / blockSkyIntensity)
    #offsetLst = [v for k,v in mosaicDoc['offsets'].iteritems()]
    offsets = np.array(offsetLst)
    relOffsets = np.array(relOffsetLst)
    return offsets.std(), relOffsets.std()



def get_total_sigma(band, semester):
    """Compute dispersion in the total (net) sky offsets for each frame."""
    fields = _get_fields(band, semester)
    sel = {"OBJECT": {"$in": fields}, "FILTER": band,
            "1.net_scalar_offset":{"$exists":1}}
            #"2.net_scalar_offset":{"$exists":1},
            #"3.net_scalar_offset":{"$exists":1},
            #"4.net_scalar_offset":{"$exists":1}
    imageLog = WIRLog()
    docs = imageLog.get(sel, ['1.net_scalar_offset'])
    offsets = []
    skyI = []
    for imageKey, doc in docs.iteritems():
        print imageKey
        offsets.append(doc['1']['net_scalar_offset'])
        skyI.append(paired_sky_intensity(imageLog, imageKey, "1"))
    offsets = np.array(offsets)
    skyIntensity = np.array(skyI)
    offsetsRel = offsets / skyIntensity
    print offsets
    print band, semester, offsets.std(), offsets.min(), offsets.max()
    return offsets.std(), offsetsRel.std()

def mean_block_sky_intensity(fieldName, band):
    """Compute the mean intensity seen during observations of this block."""
    imageLog = WIRLog()
    print "mean_block_sky_intensity", fieldName
    imageKeys = imageLog.search({"OBJECT": fieldName, "FILTER": band,
        "1.wirpsfphot_windowzp":{"$exists":1}})
    skyI = []
    for imageKey in imageKeys:
        skyI.append(paired_sky_intensity(imageLog, imageKey, "1"))
    skyI = np.array(skyI)
    return np.mean(skyI)

def paired_sky_intensity(imageLog, imageKey, ext):
    """Gets the median sky level from the paired sky sample for the given image."""
    zpKey = ext+".wirpsfphot_windowzp"
    aduKey = ext+".median_sky_level"
    diskDoc = imageLog.collection.find_one({"_id":imageKey},
            ["EXPTIME",zpKey,"median_sky_source"])
    skyDoc = imageLog.collection.find_one({"_id":diskDoc["median_sky_source"]},
            [aduKey])
    skyI = skyDoc[ext]['median_sky_level'] \
            * 10.**(-0.4*diskDoc[ext]['wirpsfphot_windowzp']) \
            / diskDoc['EXPTIME']
    return skyI


def _get_fields(band, semester):
    fieldSel = {"FILTER":band, "OBJECT":{"$nin":["M31-44","M31-45","M31-46"]}}
    if semester == "2007B":
        fieldSel['RUNID'] = {"$in": ["07BC20","07BH47"]}
    else:
        fieldSel['RUNID'] = {"$in": ["09BC29","09BH52"]}
    imageLog = WIRLog()
    fields = imageLog.find_unique("OBJECT", selector=fieldSel)
    return fields

if __name__ == '__main__':
    main()
