import os
import shutil
import sys

import pymongo
import numpy as np
import pyfits
import pywcs
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib as mpl
import scipy.ndimage

from owl.Footprint import equatorialToTan
import moastro.astromatic

import mosaicdb
import difftools
from multisimplex import SimplexScalarOffsetSolver
import offsettools
import blockmosaic
from blockdb import BlockDB

# from scalar_hierarchy import mean_block_sky_intensity
from andpipe.wircam.skyanalysis.mean_sky_level import mean_level_dn
from andpipe.wircam.calibset.mosaic.offsetratiomap import FieldLevelPlotter
from andpipe.skysubpub.diffmaps import two_image_grid


def test():
    # make_bootstrap("Ks")
    # make_bootstrap("J")
    # make_sbdiff_mosaic("J")
    # make_sbdiff_mosaic("Ks")
    mosaicDB = mosaicdb.MosaicDB(dbname="m31", cname="mosaics")
    rmsPlot = BootstrapRMSPlot(mosaicDB)
    rmsPlot.plot("skysubpub/bootstrap_sb_rms")

    #plot_recovery_hists()
    recoveryPlotter = OffsetRecoveryHists()
    recoveryPlotter.plot("skysubpub/bootstrap_recovery_hist")
    
    # Map of the sigma in the expected - realized sky offsets for each block
    recoveryFieldmap = OffsetRecoveryFieldMap()
    recoveryFieldmap.plot("skysubpub/bootstrap_recovery_map")


def make_bootstrap(band, mosaicName="fw100k_scalar_medsky_0709"):
    """docstring for make_bootstrap"""
    mosaicDB = mosaicdb.MosaicDB(dbname="m31", cname="mosaics")
    bootstrapper = ScalarBootstrap(mosaicDB)
    bootstrapper.run_bootstrap("%s_%s" % (mosaicName, band),
            nTrials=100, startIndex=0)


def make_sbdiff_mosaic(band, setName="fw100k_medsky",
        mosaicRootName="fw100k_scalar_medsky_0709"):
    """docstring for make_sbdiff_mosaic"""
    mosaicName = "%s_%s" % (mosaicRootName, band)
    mosaicDB = mosaicdb.MosaicDB(dbname="m31", cname="mosaics")
    blockDB = BlockDB()
    bmosaics = BootstrappedMosaics(mosaicDB, mosaicName, band, setName,
            blockDB,
            "wircam/%s/mosaics/bootstrap/%s_%s" % (setName, mosaicRootName,
                band),
            pixScale=10.)
    bmosaics.make_all()
    bmosaics.make_sb_diffs()
    bmosaics.make_sb_diff_rms()


class ScalarBootstrap(object):
    """Bootstraps mosaic construction with scalar sky offsets.
    
    The bootstraps are stored in the mosaic's document, as a 'bootstrap'
    sub-document.

    Solvers for the bootstrap iterations are stored in the "skyoffsets"
    DB
    """
    def __init__(self, mosaicDB):
        super(ScalarBootstrap, self).__init__()
        self.mosaicDB = mosaicDB
        
    def run_bootstrap(self, mosaicName, nTrials=1000,
            dbname="skyoffsets", url="localhost", port=27017, startIndex=0):
        """Performs a bootstrap computation of scalar sky offsets.
        
        dbname and specifies the database where solvers for each bootstrap
        are stored.

        Each bootstrap iteration creates a SimplexScalarOffsetSolver, whose
        solvers put solution documents in the collection named mosaicName+i
        where i in the bootstrap iteration number.
        """
        self.dbname = dbname
        self.url, self.port = url, port
        
        connection = pymongo.Connection(self.url, self.port)
        self.solverDB = connection[self.dbname]
        
        logPath = os.path.join("skyoffsets", "bootstrap.log")

        mosaicDoc = self.mosaicDB.get_mosaic_doc(mosaicName)
        if 'bootstrap' in mosaicDoc and startIndex == 0:
            # remove existing bootstrap document
            self.mosaicDB.collection.update({"_id": mosaicName},
                    {"$unset": {"bootstrap": 1}})

        couplings = difftools.Couplings.load_doc(mosaicDoc['couplings'])
        offsets = mosaicDoc['offsets']
        
        print "Bootstrapping",
        for i in xrange(startIndex, nTrials):
            sys.stdout.write("Iteration: %d / %i \r" % (i, nTrials))
            sys.stdout.flush()
            solverCName = mosaicName + "_boot_%i" % i
            self._run_trial(mosaicName, couplings, offsets, solverCName,
                    logPath)
            # Delete teh solver collection
            self._delete_solver(solverCName)
        print "Bootstrap complete"

    def _run_trial(self, mosaicName, origCouplings, offsets, solverCName,
            logPath):
        """Runs a single bootstrap trial.

        Results of the bootstrap (
        
        :param origCouplings: the couplings instance of the original mosaic
        :param offsets: dictionary of scalar offsetes for each block
        :param solverCName: name of collection where solver documents
            are stored.
        """
        fieldList = []
        origOffsetList = []
        for field, offset in offsets.iteritems():
            fieldList.append(field)
            origOffsetList.append(offset)
        nFields = len(fieldList)

        # For each field, choose at random a residual from the population
        # or residuals (origOffsetList).
        ind = np.random.random_integers(0, high=nFields - 1, size=nFields)
        perturbations = {}
        for i, field in zip(ind, fieldList):
            perturbations[field] = origOffsetList[i]
        
        # Create a Couplings instance using resampled sky errors
        # embedded into the field diffs
        resampCouplings = ResampledCouplings(origCouplings, offsets,
                perturbations)

        # Scale of simplex starting point dispersions
        initScale = 5.
        restartScale = 2.
        diffList = np.array(origOffsetList)
        diffSigma = diffList.std()
        initSigma = initScale * diffSigma
        restartSigma = restartScale * diffSigma

        # Solve the sky offsets
        solver = SimplexScalarOffsetSolver(dbname=self.dbname,
                cname=solverCName, url=self.url, port=self.port)
        solver.resetdb()
        solver.multi_start(resampCouplings, 8, logPath, cython=True,
                mp=True,
                initSigma=initSigma,
                restartSigma=restartSigma)
        resampOffsets = solver.find_best_offsets()

        # Append results to the mosaic document
        d = {}
        for field, offset in resampOffsets.iteritems():
            pert = perturbations[field]
            d["bootstrap." + field + ".perturbation"] = pert
            d["bootstrap." + field + ".correction"] = offset
        updateDoc = {"$push": d}
        self.mosaicDB.collection.update({"_id": mosaicName}, updateDoc)

    def _delete_solver(self, solverCName):
        """Drops the solver collection used for a single bootstrap iteration.
        """
        self.solverDB.drop_collection(solverCName)


class ResampledCouplings(difftools.Couplings):
    """Couplings subclass that modifies an original Couplings instance
    to simulate the effect of random errors on the sky backgrounds.
    
    Couplings stores the difference between coupled fields as

        field1, field2: field1 - field2 = diff
    
    The original mosaic has sky offsets so that the true (minimized)
    difference between fields is

        field1-offset1 - (field2-offset2)
        = diff - offset1 + offset2

    Then given the perturbations bootstrapped for each field (pert1... pertN)
    the new difference is, after correcting for original offsets,

        field1, field2: (field1-offset1+resid1) - (field2-offset2+resid2)
                        = diff - offset1 + offset2 + resid1 - resid2

    Thus, the residuals are seen as *additional* sky flux added to each image.
    """
    def __init__(self, origCouplings, origOffsets, perturbations):
        super(ResampledCouplings, self).__init__()
        self.origCouplings = origCouplings
        self.origOffsets = origOffsets
        self.perturbations = perturbations
        
        self.fieldDiffs = {}
        for (field1, field2), diff in origCouplings.fieldDiffs.iteritems():
            self.fieldDiffs[(field1, field2)] = diff \
                    - self.origOffsets[field1] \
                    + self.origOffsets[field2] \
                    + self.perturbations[field1] \
                    - self.perturbations[field2]
        self.fieldDiffSigmas = origCouplings.fieldDiffSigmas
        self.fieldDiffAreas = origCouplings.fieldDiffAreas
        self.fieldLevels = origCouplings.fieldLevels
        self.fields = origCouplings.fields


class OffsetRecoveryHists(object):
    """Plot histograms of the error in recovering expected offsets."""
    def __init__(self, mosaicRootName="fw100k_scalar_medsky_0709",
            setName="fw100k_medsky"):
        super(OffsetRecoveryHists, self).__init__()
        self.setName = setName
        self.mosaicRootName = mosaicRootName
        self.mosaicDB = mosaicdb.MosaicDB(dbname="m31", cname="mosaics")
    
    def plot(self, plotPath):
        """docstring for plot"""
        scalarJRecovery = np.array(self._get_offset_recovery_dist("J"))
        scalarKRecovery = np.array(self._get_offset_recovery_dist("Ks"))
        bins = np.arange(-0.5, 0.5, 0.01)

        fig = plt.figure(figsize=(3.5, 2.5))
        ax = fig.add_axes((0.18, 0.15, 0.75, 0.8))
        ax.hist(scalarJRecovery, bins=bins, color='b', hatch='/',
                histtype='step',
                label=r"$J$, $\sigma=%.2f$" % scalarJRecovery.std())
        ax.hist(scalarKRecovery, bins=bins, color='k', hatch='\\',
                histtype='step',
                label=r"$K_s$, $\sigma=%.2f$" % scalarKRecovery.std())
        ax.legend()
        ax.set_xlabel(r"Expected - Realized Sky Offset (\% of sky)")
        ax.set_ylabel(r"Bootstrap Blocks")
        fig.savefig(plotPath + ".pdf", format="pdf")
        fig.savefig(plotPath + ".eps", format="eps")

    def _get_offset_recovery_dist(self, band):
        """return list of expected-true offsets, as a percent of sky level"""
        mosaicName = "%s_%s" % (self.mosaicRootName, band)
        doc = self.mosaicDB.get_mosaic_doc(mosaicName)
        # use scalar_hierarchy module for code that gets the mean sky intensity
        # while observing a block
        allRelOffsets = []
        fieldnames = doc['bootstrap'].keys()
        for fieldname in fieldnames:
            field, band = fieldname.split("_")[:2]
            blockSky, blockSkySigma = mean_level_dn(self.setName, band)
            corrections = np.array(doc['bootstrap'][fieldname]['correction'])
            perturbations = np.array(
                doc['bootstrap'][fieldname]['perturbation'])
            netRelOffsets = (corrections - perturbations) / blockSky * 100.
            allRelOffsets += netRelOffsets.tolist()
        return allRelOffsets


class OffsetRecoveryFieldMap(object):
    """Plot the standard deviation of expected vs realized offset at the
    locations of blocks"""
    def __init__(self, mosaicRootName="fw100k_scalar_medsky_0709",
            setName="fw100k_medsky"):
        super(OffsetRecoveryFieldMap, self).__init__()
        self.mosaicRootName = mosaicRootName
        self.setName = setName
        self.mosaicDB = mosaicdb.MosaicDB(dbname="m31", cname="mosaics")

    def plot(self, plotPath):
        fig, grid = two_image_grid()

        self._plot_band("J", grid[0], grid.cbar_axes[0])
                # cticks=np.arange(0.05, 0.11, 0.01))
        self._plot_band("Ks", grid[1], grid.cbar_axes[1])
                # cticks=np.arange(0.05, 0.12, 0.01))

        grid[0].set_ylabel(r"$\eta$ (degrees)")
        grid[0].set_xlabel(r"$\xi$ (degrees)")
        grid[1].set_xlabel(r"$\xi$ (degrees)")

        grid[0].text(0.9, 0.9, r"$J$", ha='right', va='top',
                transform=grid[0].transAxes)
        grid[1].text(0.9, 0.9, r"$K_s$", ha='right', va='top',
                transform=grid[1].transAxes)

        fig.savefig(plotPath + ".pdf", format="pdf")

    def _plot_band(self, band, ax, cbar, cticks=None):
        fieldRecoverySigma = self._compute_field_recovery_sigma(band)
        clabel = r"$\sigma(\mathrm{expect.-obs. ~offset})$ [\% sky]"
        plotter = FieldLevelPlotter(ax,
                {"instrument": "WIRCam", "kind": "ph2"})
        plotter.set_field_values(fieldRecoverySigma)
        plotter.render(cbar=True, cax=cbar, cbarOrientation="horizontal",
                clabel=clabel, cticks=cticks,
                borderColour='k')
        cbar.axis['top'].toggle(ticklabels=True, label=True)
        cbar.axis['top'].set_label(clabel)
        ax.set_xlim(1.1, -1.1)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')

    def _compute_field_recovery_sigma(self, band):
        """For each field, find the standard deviation of expected - obs.
        offsets. Pack the result in a field dictionary.
        """
        mosaicName = "%s_%s" % (self.mosaicRootName, band)
        doc = self.mosaicDB.get_mosaic_doc(mosaicName)
        fieldSigmas = {}
        fieldnames = doc['bootstrap'].keys()
        for fieldname in fieldnames:
            print fieldname
            field, band = fieldname.split("_")[:2]
            # blockSky = mean_block_sky_intensity(field, band)
            blockSky, blockSkySigma = mean_level_dn(self.setName, band)
            corrections = np.array(doc['bootstrap'][fieldname]['correction'])
            perturbations = np.array(
                doc['bootstrap'][fieldname]['perturbation'])
            netRelOffsets = (corrections - perturbations) / blockSky * 100.
            fieldSigmas[field] = netRelOffsets.std()
            print field, band, netRelOffsets.std()
        return fieldSigmas


class BootstrappedMosaics(object):
    """Generate bootstrapped mosaics at 10 arcsec/pixel resolution."""
    def __init__(self, mosaicDB, mosaicName, band, setName, blockDB, workDir,
            pixScale=10.):
        super(BootstrappedMosaics, self).__init__()
        self.mosaicDB, self.mosaicName = mosaicDB, mosaicName
        self.blockDB = blockDB
        self.pixScale = pixScale
        self.workDir = workDir
        self.band = band
        self.setName = setName  # calibrated set
        
        # Store the bootstrap data document in self.b
        doc = self.mosaicDB.get_mosaic_doc(mosaicName)
        self.b = doc['bootstrap']
        self.couplings = difftools.Couplings.load_doc(doc['couplings'])

        # subsampled block paths
        self.subsampledDocs = {}

    def make_all(self):
        """Make mosaics from all bootstrap instances.
        
        This method also caches paths to subsampled mosaics in the bootstrap
        document.
        """
        self._subsample_blocks(self.pixScale)
        self._make_ref_mosaic()
        sampleField = self.b.keys()[0]
        nBoots = len(self.b[sampleField]['perturbation'])
        bootIDs = range(nBoots)
        print "running bootstrap IDs:"
        print bootIDs
        for bootID in bootIDs:
            downsampledPath = self._make_mosaic(bootID)
            self.mosaicDB.collection.update({"_id": self.mosaicName},
                {"$set": {"bootstrapmosaics.%i" % bootID: downsampledPath}})

    def _subsample_blocks(self, pixScale):
        """Subsample the mosaic blocs to the given pixel scale in arcsec/pix
        so that perturbations and mosaics are made on the small images."""
        if os.path.exists(self.workDir) is False: os.makedirs(self.workDir)
        resampDir = os.path.join(self.workDir, "resamp")
        if os.path.exists(resampDir) is False: os.makedirs(resampDir)
        fieldnames = self.b.keys()  # names of blocks
        print "Working with fields", fieldnames
        # blockDocs = self.blockDB.find_blocks({"_id": {"$in": fieldnames}})
        blockDocs = self.blockDB.find_blocks({"calibrated_set": self.setName,
            "FILTER": self.band})
        print "type(blockDocs)", type(blockDocs)
        print "blockDocs.keys()", blockDocs.keys()
        imagePathList = [blockDocs[k]['image_path'] for k in fieldnames]
        weightPathList = [blockDocs[k]['weight_path'] for k in fieldnames]
        swarpConfigs = {"WEIGHT_TYPE": "MAP_WEIGHT",
                "SUBTRACT_BACK": "N",
            "RESAMPLE_DIR": resampDir,
            "COMBINE_TYPE": "WEIGHTED",
            "RESAMPLE": "Y", "COMBINE": "N",
            "PIXEL_SCALE": "%.2f" % pixScale, "PIXELSCALE_TYPE": "MANUAL"}
        swarp = moastro.astromatic.Swarp(imagePathList, "resample",
            weightPaths=weightPathList, configs=swarpConfigs,
            workDir=resampDir)
        swarp.run()
        for imagePath, weightPath, k in zip(imagePathList, weightPathList,
                fieldnames):
            basename = os.path.splitext(os.path.basename(imagePath))[0]
            self.subsampledDocs[k] = {"_id": k,
                "image_path": os.path.join(resampDir,
                    basename + ".resamp.fits"),
                "weight_path": os.path.join(resampDir,
                    basename + ".resamp.weight.fits")}
    
    def _make_ref_mosaic(self):
        """Compute the original mosaic from the subsampled blocks. This acts
        as a reference surface."""
        mosaicDoc = self.mosaicDB.get_mosaic_doc(self.mosaicName)
        origOffsets = mosaicDoc['offsets']
        mosaicName = "ref_mosaic"
        rescaledOrigOffsets = {}
        for k, delta in origOffsets.iteritems():
            rescaledOrigOffsets[k] = delta * (self.pixScale / 0.3) ** 2.
        mosaicPath, weightPath = blockmosaic.block_mosaic(self.subsampledDocs,
                rescaledOrigOffsets, mosaicName,
                self.workDir, offset_fcn=offsettools.apply_offset)
        shutil.rmtree(os.path.join(self.workDir, "offset_images"))
        # Store in the MosaicDB
        self.mosaicDB.collection.update({"_id": self.mosaicName},
            {"$set": {"ref_path": mosaicPath, "ref_weight_path": weightPath}})

    def _make_mosaic(self, bootIndex):
        """Make a reconstructed, subsampled, mosaic from bootstrap instance
        bootID."""
        stackDir = os.path.join(self.workDir, "realizations")
        if os.path.exists(stackDir) is False: os.makedirs(stackDir)
        mosaicDoc = self.mosaicDB.get_mosaic_doc(self.mosaicName)
        origOffsets = mosaicDoc['offsets']
        pertDict = self._get_perturbation_dict(bootIndex, "perturbation")
        corrDict = self._get_perturbation_dict(bootIndex, "correction")
        netOffsets = {}
        for field in self.b.keys():
            # The net offset, include original solution, perturbation
            # and correction
            netOffset = origOffsets[field] - pertDict[field] + corrDict[field]
            # Rescale for new pixel scale, original pixel scale was
            # 0.3 arcsec/pixel
            netOffsets[field] = netOffset * (self.pixScale / 0.3) ** 2.
        mosaicName = self.mosaicName + "_b%i" % bootIndex
        mosaicPath, weightPath = blockmosaic.block_mosaic(
            self.subsampledDocs, netOffsets, mosaicName,
            stackDir, offset_fcn=offsettools.apply_offset)
        shutil.rmtree(os.path.join(stackDir, "offset_images"))
        return mosaicPath

    def _get_perturbation_dict(self, i, key):
        """Reduces the bootstrap document by getting the perturbation at index
        `i` for each field under the `key` (ie `perturbation` or `correction`).
        """
        perts = {}
        for field in self.b.keys():
            perts[field] = self.b[field][key][i]
        return perts

    def make_sb_diffs(self):
        """Compute the surface brightness of each realization, and difference
        against the reference image. Resulting image paths are stored
        in the MosaicDB under sb_diff_paths."""
        sb = lambda im: -2.5 * np.log10(im / self.pixScale ** 2.)
        sbDiffDir = os.path.join(self.workDir, "bootstrap_sbdiffs")
        if os.path.exists(sbDiffDir) is False: os.makedirs(sbDiffDir)
        # First make the SB reference image.
        bDoc = self.mosaicDB.get_mosaic_doc(self.mosaicName)
        refImagePath = bDoc['ref_path']
        refFITS = pyfits.open(refImagePath)
        refImage = refFITS[0].data
        refSBImage = sb(refImage)
        for bootID, bootPath in bDoc['bootstrapmosaics'].iteritems():
            sbDiffPath = os.path.join(sbDiffDir, str(bootID) + ".fits")
            sbDiffWeightPath = os.path.join(sbDiffDir,
                    str(bootID) + ".weight.fits")
            bootFITS = pyfits.open(bootPath)
            sbImage = sb(bootFITS[0].data)
            sbDiffImage = sbImage - refSBImage
            weightImage = np.zeros(sbImage.shape)
            weightImage[np.isfinite(sbDiffImage)] = 1.
            pyfits.writeto(sbDiffPath, sbDiffImage, refFITS[0].header,
                    clobber=True)
            pyfits.writeto(sbDiffWeightPath, weightImage, refFITS[0].header,
                    clobber=True)
            bootFITS.close()
            self.mosaicDB.collection.update({"_id": self.mosaicName},
                {"$set": {"bootstrap_sbdiffs.%i" % int(bootID): sbDiffPath,
                "bootstrap_sbdiff_weights.%i" % int(bootID): sbDiffWeightPath}})
        refFITS.close()

    def make_sb_diff_rms(self):
        """Make an RMS map of surface brightness deviations from the
        reference image.
        """
        bDoc = self.mosaicDB.get_mosaic_doc(self.mosaicName)
        keys = bDoc['bootstrap_sbdiffs'].keys()
        imagePathList = [bDoc['bootstrap_sbdiffs'][k] for k in keys]
        weightPathList = [bDoc['bootstrap_sbdiff_weights'][k] for k in keys]

        swarpConfigs = {"WEIGHT_TYPE": "MAP_WEIGHT",
            "SUBTRACT_BACK": "N",
            "COMBINE_TYPE": "CHI2",
            "RESAMPLE": "N", "COMBINE": "Y",
            "PIXEL_SCALE": "%.2f" % self.pixScale, "PIXELSCALE_TYPE": "MANUAL"}
        swarp = moastro.astromatic.Swarp(imagePathList, "sb_rms",
            weightPaths=weightPathList, configs=swarpConfigs,
            workDir=self.workDir)
        swarp.run()
        mosaicPath, weightPath = swarp.mosaic_paths()
        self.mosaicDB.collection.update({"_id": self.mosaicName},
                    {"$set": {"bootstrap_sbrms": mosaicPath,
                    "bootstrap_sbrms_weight": weightPath}})


class BootstrapRMSPlot(object):
    """Plot the RMS maps of surface brightness seen in the boostraps"""
    def __init__(self, mosaicDB, mosaicRootName="fw100k_scalar_medsky_0709"):
        super(BootstrapRMSPlot, self).__init__()
        self.mosaicRootName = mosaicRootName
        self.mosaicDB = mosaicDB
    
    def plot(self, plotPath):
        """docstring for plot"""
        fig, grid = two_image_grid()

        self._plot_band(grid[0], grid.cbar_axes[0], "J", clim=(0., 1.5),
                cticks=[0., 0.5, 1., 1.5])
        self._plot_band(grid[1], grid.cbar_axes[1], "Ks", clim=(0., 1.5),
                cticks=[0.5, 1., 1.5])

        grid[0].set_ylabel(r"$\eta$ (degrees)")
        grid[0].set_xlabel(r"$\xi$ (degrees)")
        grid[1].set_xlabel(r"$\xi$ (degrees)")

        grid[0].text(0.9, 0.9, r"$J$", ha='right', va='top',
                transform=grid[0].transAxes)
        grid[1].text(0.9, 0.9, r"$K_s$", ha='right', va='top',
                transform=grid[1].transAxes)

        fig.savefig(plotPath + ".pdf", format="pdf")
        fig.savefig(plotPath + ".eps", format="eps")

    def _plot_band(self, ax, cbar, band, clim=(0., 1.), cticks=None):
        mosaicName = "%s_%s" % (self.mosaicRootName, band)
        doc = self.mosaicDB.get_mosaic_doc(mosaicName)
        rmsPath = doc['bootstrap_sbrms']
        fits = pyfits.open(rmsPath)
        image = fits[0].data
        header = fits[0].header
        image[image > 5.] = np.nan
        smoothImage = scipy.ndimage.filters.gaussian_filter(image, 5)
        extent = self._get_extent(header)
        norm = mpl.colors.Normalize(clip=True, vmin=min(clim), vmax=max(clim))
        im = ax.imshow(image, extent=extent, origin='lower',
            cmap=mpl.cm.jet, norm=norm)
        ax.contour(smoothImage, [0.05, 0.1, 0.2], extent=extent,
                origin='lower', linestyles=['solid', 'dashed', 'dashdot'],
                colors='w', antialiased=True)
        cb = plt.colorbar(im, cax=cbar, ax=ax, orientation='horizontal')
        cbar.axis['top'].toggle(ticklabels=True, label=True)
        cbar.axis['top'].set_label(r"RMS (mag $\Box^{-2}$)")
        if cticks is not None:
            cb.set_ticks(cticks)
        ax.set_xlabel(r"$\xi$ (degrees)")
        ax.set_ylabel(r"$\eta$ (degrees)")
        fits.close()

    def _get_extent(self, header):
        """Get extent of the image, in tangent plane coordinates
        # left, right, bottom, top
        """
        wcs = pywcs.WCS(header=header)
        xsize = header['NAXIS1']
        ysize = header['NAXIS2']
        x = [0., xsize]
        y = [0., ysize]
        p = wcs.all_pix2sky(x, y, 0)
        raLL, raUR = p[0][0], p[0][1]
        decLL, decUR = p[1][0], p[1][1]

        ra0 = 10.6846833
        dec0 = 41.2690361
        xiMax, etaMin = equatorialToTan(raLL, decLL, ra0, dec0)
        xiMin, etaMax = equatorialToTan(raUR, decUR, ra0, dec0)

        return (xiMax, xiMin, etaMin, etaMax)

if __name__ == '__main__':
    test()