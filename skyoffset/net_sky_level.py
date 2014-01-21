"""Computes the sky level of each image (after scalar offset) and captures
the typical sky level seen during the campaign
"""
import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from andpipe.imagelog import WIRLog
from mosaicdb import MosaicDB

MU_J = 19.3 # mag/arcsec^2
MU_KS = 16.60

def main():
    """docstring for main"""
    plot_sky_sb_histograms()
    #median_sky_levels()
    campaign_sky_levels()

def plot_sky_sb_histograms():
    plotPath = "skysubpub/sky_level_hist"

    skyJ7SB = get_sky_field_levels({"FILTER": "J", "TYPE":"sky",
        "RUNID":{"$in":["07BC20","07BH47"]}})
    skyJ9SB = get_sky_field_levels({"FILTER": "J", "TYPE":"sky",
        "RUNID":{"$in":["09BC29","09BH52"]}})
    skyK7SB = get_sky_field_levels({"FILTER": "Ks", "TYPE":"sky",
        "RUNID":{"$in":["07BC20","07BH47"]}})
    skyK9SB = get_sky_field_levels({"FILTER": "Ks", "TYPE":"sky",
        "RUNID":{"$in":["09BC29","09BH52"]}})

    fig = plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.14, bottom=0.12, right=0.95, top=0.98,
        wspace=None, hspace=0.3)
    axJ = fig.add_subplot(211)
    axK = fig.add_subplot(212)

    print "found %i J7 sky" % len(skyJ7SB)
    print skyJ7SB

    #axJ07.plot(skyJ7SB, skyJ7SB)
    bins = np.arange(12.5,16.,0.05)
    print bins
    axJ.hist(skyJ7SB, bins=bins, histtype='step', ec='r', label="2007B J",zorder=10)
    axJ.hist(skyJ9SB, bins=bins, histtype='stepfilled', ec='k', fc=(0.8,0.8,0.8,0.5),
            ls='dashdot', label="2009B J")
    axJ.set_xlabel(r"Sky Levels $\mu_J$ (mag arcsec$^{-2}$)")
    axJ.set_ylabel(r"$N$ sky obs.")
    axJ.set_xlim(14.2,15.4)
    axK.hist(skyK7SB, bins=bins, histtype='step', ec='r', label="2007B K",zorder=10)
    axK.hist(skyK9SB, bins=bins, histtype='stepfilled', ec='k', fc=(0.8,0.8,0.8,0.5),
            ls='dashdot', label="2009B K")
    axK.set_xlim(13.,13.8)
    axK.set_xlabel(r"Sky Levels $\mu_{K_s}$ (mag arcsec$^{-2}$)")
    axK.set_ylabel(r"$N$ sky obs.")

    fig.savefig(plotPath+".pdf", format="pdf")

def median_sky_levels():
    """docstring for median_sky_levels"""
    skyJ7SB = get_sky_field_levels({"FILTER": "J", "TYPE":"sky",
        "RUNID":{"$in":["07BC20","07BH47"]}})
    medianJ = np.median(skyJ7SB)

    skyKSB = get_sky_field_levels({"FILTER": "Ks", "TYPE":"sky"})
    medianK = np.median(skyKSB)

    print "Median J: %.2f" % medianJ
    print "Median K: %.2f" % medianK

def campaign_sky_levels():
    """Compute median and 95 percentile sky levels for each WIRCam run.
    Note that 2009B had a bimodal sky distribution."""
    skyJ7SB = get_sky_field_levels({"FILTER": "J", "TYPE":"sky",
        "RUNID":{"$in":["07BC20","07BH47"]}})
    skyJ9SB = get_sky_field_levels({"FILTER": "J", "TYPE":"sky",
        "RUNID":{"$in":["09BC29","09BH52"]}})
    skyK7SB = get_sky_field_levels({"FILTER": "Ks", "TYPE":"sky",
        "RUNID":{"$in":["07BC20","07BH47"]}})
    skyK9SB = get_sky_field_levels({"FILTER": "Ks", "TYPE":"sky",
        "RUNID":{"$in":["09BC29","09BH52"]}})
    
    brightSkyJ9SB = skyJ9SB[skyJ9SB > 15.]
    dimSkyJ9SB = skyJ9SB[skyJ9SB < 15.]

    print "Median and 95% C.I."

    print_ci("2007B J", skyJ7SB)
    print_ci("2009B J dim", dimSkyJ9SB)
    print_ci("2009B J bright", brightSkyJ9SB)

    print_ci("2007B Ks", skyK7SB)
    print_ci("2009B Ks", skyK9SB)
    #print "J. %.2f (%.2f,%.2f)" % (np.median(skyJ7SB),)

def print_ci(name, array):
    """docstring for print_ci"""
    print name,
    print np.median(array), ";",
    print scipy.stats.scoreatpercentile(array, 2.5, limit=()), "--",
    print scipy.stats.scoreatpercentile(array, 97.5, limit=())

def get_sky_field_levels(selector):
    """Returns numpy array of sky levels (in surface brightness untis)
    seen in the sky fields.
    """
    imageLog = WIRLog()
    dataFields = ["1.median_sky_level","1.wirpsfphot_windowzp",
            "2.median_sky_level","2.wirpsfphot_windowzp",
            "3.median_sky_level","3.wirpsfphot_windowzp",
            "4.median_sky_level","4.wirpsfphot_windowzp",
            "EXPTIME"]
    docs = imageLog.get(selector, dataFields)
    print "Found %i images" % len(docs.keys())
    sbLst = []
    area = 0.3*0.3 # arcsec^2/pix
    for imageKey, doc in docs.iteritems():
        try:
            level = doc["1"]["median_sky_level"]
            exptime = doc["EXPTIME"]
            zp = doc["1"]["wirpsfphot_windowzp"]
            I = level * 10.**(-2.*zp/5.) / area / exptime
            sb = -2.5*math.log10(I)
            sbLst.append(sb)
        except:
            continue
    sbLst = np.array(sbLst)
    print "sbLst", sbLst
    return sbLst

if __name__ == '__main__':
    main()

