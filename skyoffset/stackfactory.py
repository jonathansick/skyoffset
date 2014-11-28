#!/usr/bin/env python
# encoding: utf-8
"""
Make stacks from a set of chip images.
"""
import os
import multiprocessing

import numpy as np
import astropy.io.fits
from astropy.stats.funcs import sigma_clip

from moastro.astromatic import Swarp

from difftools import SliceableImage
from difftools import ResampledWCS
from difftools import Overlap
from offsettools import apply_offset
from noisefactory import NoiseMapFactory


class ChipStacker(object):
    """General-purpose class for stacking single-extension FITS, adding a
    sky offset to achieve uniform sky bias.

    Parameters
    ----------
    stackdb : :class:`skyoffset.imagedb.MosaicDB` instance
        The MosaicDB instance to store stack documents in.
    workdir : str
        Directory to make stacks in. This directory will be created if
        necessary.
    swarp_configs : dict
        A dictionary of configurations to pass to
        :class:`moastro.astromatic.Swarp`.
    """
    def __init__(self, stackdb, workdir, swarp_configs=None):
        super(ChipStacker, self).__init__()
        self.workdir = workdir
        self.noise_path = None
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        self.stackdb = stackdb
        if swarp_configs:
            self._swarp_configs = dict(swarp_configs)
        else:
            self._swarp_configs = {}

    def pipeline(self, stack_name, image_keys, image_paths, weight_paths,
                 noise_paths=None, db_meta=None, convergence_tol=1e-4,
                 n_iter_max=4):
        """Pipeline for running the ChipStacker method to produce stacks
        and addd them to the stack DB.

        Parameters
        ----------
        stack_name : str
            Name of the stack being produced (for filenames and MongoDB
            document ``_id``s.
        image_keys : list
            List of image identifier strings (can be image keys from your
            :class:`moastro.imagelog.ImageLog`.
        image_paths : list
            List of paths (strings) to images being stacked.
        weight_paths : list
            List of paths (strings) to weight maps, corresponding to
            ``image_paths``.
        noise_paths : list
            List of paths (strings) to noise maps, corresponding to
            ``iamge_paths``.
        dbmeta : dict
            Arbitrary metadata to store in the stack's StackDB document.
        convergence_tol : float
            Fractional converence tolerance to halt frame offset estimation.
        """
        im_paths = dict(zip(image_keys, image_paths))
        w_paths = dict(zip(image_keys, weight_paths))
        self._stack_images(image_keys, im_paths, w_paths, stack_name,
                           convergence_tol, n_iter_max)
        self._remove_offset_frames()
        self._renormalize_weight()
        if noise_paths:
            self._make_noisemap(image_keys, noise_paths,
                                [w_paths[ik] for ik in image_keys],
                                self._coadd_path)
        stack_doc = {"_id": stack_name,
                     "image_path": self._coadd_path,
                     "weight_path": self._coadd_weightpath,
                     "offsets": {ik: float(offset) for ik, offset
                                 in self.offsets.iteritems()}}
        if self.noise_path:
            stack_doc.update({"noise_path": self.noise_path})
        if db_meta is not None:
            stack_doc.update(db_meta)
        self.stackdb.c.save(stack_doc)

        # Define the stack's footprint
        self.stackdb.add_footprint_from_header(
            stack_name,
            astropy.io.fits.getheader(self._coadd_path))

    def _stack_images(self, image_keys, image_paths, weight_paths, stack_name,
                      convergence_tol, n_iter_max):
        """Make a stack with the given list of images.
        :param image_keys: list of strings identifying the listed image paths.
        :param image_paths: dict of paths to single-extension FITS image files.
        :param weight_paths: dict of paths to weight images for `image_paths`.
        :return: path to stacked image.
        """
        self.image_keys = image_keys
        self.image_paths = image_paths
        self.weight_paths = weight_paths
        self.stack_name = stack_name

        # Start with the original images
        self._current_offset_paths = dict(image_paths)

        # Make resampled frames
        self.image_frames = {}
        for image_key, image_path in self.image_paths.iteritems():
            header = astropy.io.fits.getheader(image_path)
            self.image_frames[image_key] = ResampledWCS(header)

        # 1. Do initial coadd
        self._coadd_path, self._coadd_weightpath, self._coadd_frame \
            = self._coadd_frames()

        # 2. Compute overlaps of frames to the coadded frame
        self.overlaps = {}
        for image_key in self.image_keys:
            self.overlaps[image_key] \
                = Overlap(self.image_frames[image_key], self._coadd_frame)

        # 3. Estimate offsets from the initial mean, and recompute the mean
        diff_data = self._compute_differences()
        offsets = self._estimate_offsets(diff_data)

        i = 0
        while i < n_iter_max:
            i += 1
            print "CHIPSTACKER ITER", i
            prev_offsets = dict(offsets)

            self._make_offset_images(prev_offsets)
            self._coadd_path, self._coadd_weightpath, self._coadd_frame \
                = self._coadd_frames()
            self.overlaps = {}
            for imageKey in self.image_keys:
                self.overlaps[imageKey] \
                    = Overlap(self.image_frames[imageKey], self._coadd_frame)

            diff_data = self._compute_differences()
            offsets = self._estimate_offsets(diff_data)

            all_conv = True
            for ik, offset in offsets.iteritems():
                prev_offset = prev_offsets[ik]
                print ik, offset, "from", prev_offset
                if np.abs((offset - prev_offset) / offset) > convergence_tol:
                    all_conv = False
            if all_conv:
                break

        # Make final stacks
        self.offsets = offsets
        self._make_offset_images(self.offsets)
        self._coadd_path, self._coadd_weightpath, self._coadd_frame \
            = self._coadd_frames()

    def _remove_offset_frames(self):
        for imageKey in self.image_keys:
            os.remove(self._current_offset_paths[imageKey])

    def _renormalize_weight(self):
        """Renormalizes the weight image of the stack."""
        fits = astropy.io.fits.open(self._coadd_weightpath)
        image = fits[0].data
        image[image > 0.] = 1.
        fits[0].data = image
        fits.writeto(self._coadd_weightpath, clobber=True)
        fits.close()

    def _coadd_frames(self):
        """Swarps images together as their arithmetic mean."""
        imagePathList = []
        weightPathList = []
        for frame in self._current_offset_paths:
            imagePathList.append(self._current_offset_paths[frame])
            weightPathList.append(self.weight_paths[frame])
        configs = dict(self._swarp_configs)
        configs.update({'RESAMPLE': 'N', 'SUBTRACT_BACK': 'N'})
        swarp = Swarp(imagePathList, self.stack_name,
                      weightPaths=weightPathList,
                      configs=configs, workDir=self.workdir)
        swarp.run()
        coaddPath, coaddWeightPath = swarp.mosaic_paths()

        coaddHeader = astropy.io.fits.getheader(coaddPath, 0)
        coaddFrame = ResampledWCS(coaddHeader)

        return coaddPath, coaddWeightPath, coaddFrame

    def _compute_differences(self):
        """Computes the deviation of individual images to the level of the
        average.
        """
        args = []
        for imageKey, overlap in self.overlaps.iteritems():
            # framePath = self.imageLog[imageKey][self.resampledKey][hdu]
            framePath = self.image_paths[imageKey]
            frameWeightPath = self.weight_paths[imageKey]
            arg = (imageKey, framePath, frameWeightPath, "coadd",
                   self._coadd_path, self._coadd_weightpath, overlap)
            args.append(arg)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = pool.map(_compute_diff, args)
        pool.terminate()
        #results = map(_compute_diff, args)
        offsets = {}
        for result in results:
            frame, coaddKey, offsetData = result
            offsets[frame] = offsetData  # look at _compute_diff() for spec
        # Normalize offsets
        net_offset = np.mean([d['diffimage_mean']
                              for ik, d in offsets.iteritems()])
        for ik, offset in offsets.iteritems():
            offsets[ik]['diffimage_mean'] = offset['diffimage_mean'] \
                - net_offset
        return offsets

    def _estimate_offsets(self, diff_data):
        """Estimate offsets based on the simple difference of taht frame to
        the coadded surface intensity.
        """
        frames = []
        offsets = []
        for frame, data in diff_data.iteritems():
            frames.append(frame)
            offsets.append(data['diffimage_mean'])
        offsets = dict(zip(frames, offsets))
        return offsets

    def _make_offset_images(self, offsets):
        """Apply the offsets to the images, and save to disk."""
        if offsets is not None:
            self._current_offset_paths = {}
            offsetDir = os.path.join(self.workdir, "offset_frames")
            if os.path.exists(offsetDir) is False:
                os.makedirs(offsetDir)

            args = []
            for imageKey in self.image_frames:
                offset = offsets[imageKey]
                origPath = self.image_paths[imageKey]
                offsetPath = os.path.join(offsetDir,
                                          os.path.basename(origPath))
                arg = (imageKey, origPath, offset, offsetPath)
                args.append(arg)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            results = pool.map(apply_offset, args)
            #results = map(apply_offset, args)
            for result in results:
                imageKey, offsetImagePath = result
                self._current_offset_paths[imageKey] = offsetImagePath
            pool.terminate()

    def _make_noisemap(self, image_keys, noise_paths, weight_paths,
                       mosaic_path):
        """Make a noise map for this coadd given noisemaps of individual
        images.
        """
        factory = NoiseMapFactory(noise_paths, weight_paths, mosaic_path,
                                  swarp_configs=dict(self._swarp_configs),
                                  delete_temps=False)
        self.noise_path = factory.map_path


def _compute_diff(arg):
    """Worker: Computes the DC offset of frame-coadd"""
    upperKey, upperPath, upperWeightPath, lowerKey, lowerPath, \
        lowerWeightPath, overlap = arg
    # print "diff between", upperKey, lowerKey
    upper = SliceableImage.makeFromFITS(upperKey, upperPath, upperWeightPath)
    upper.setRange(overlap.upSlice)
    lower = SliceableImage.makeFromFITS(lowerKey, lowerPath, lowerWeightPath)
    lower.setRange(overlap.loSlice)
    goodPix = np.where((upper.weight > 0.) & (lower.weight > 0.))
    nPixels = len(goodPix[0])
    if nPixels > 10:
        diff_pixels = upper.image[goodPix] - lower.image[goodPix]
        diff_pixels = diff_pixels[np.isfinite(diff_pixels)]
        clipped = sigma_clip(diff_pixels, sig=5., iters=1,
                             varfunc=np.nanvar)
        median = np.nanmedian(clipped[~clipped.mask])
        sigma = np.nanstd(clipped[~clipped.mask])
        diffData = {"diffimage_mean": median,
                    "diffimage_sigma": sigma,
                    "area": nPixels}
    else:
        diffData = None
    return upperKey, lowerKey, diffData
