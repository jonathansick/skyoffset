#!/usr/bin/env python
# encoding: utf-8
"""
The resampler module provides tools for resampling a set of mosaics, both
to have matching pixel grids, and to alter the pixel scale while propagating
noise maps.
"""

import os

import numpy as np
import astropy.io.fits
import moastro.astromatic


class MosaicResampler(object):
    """Resamples a set of mosaics to a common frame.
    
    Parameters
    ----------
    workdir : str
        Directory to make resampled mosaics in.
    mosaicdb : :class:`skyoffset.imagedb.MosaicDB`
        Optional, a MosaicDB to store the resampled mosaics in.
    target_fits : str
        (Optional) Path to a FITS image that defines the desired target
        frame for the mosaics.
    """
    def __init__(self, workdir, mosaicdb=None, target_fits=None):
        super(MosaicResampler, self).__init__()
        self.mosaicdb = mosaicdb

        self._mosaic_cursors = []
        self._target_fits = target_fits

        self.workdir = workdir
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

    def add_mosaics(self, docs):
        """Add mosaic(s) to resample.

        The *first* mosaic added will establish the target frame. Thus it
        should be the image with the largest footprint.
        
        Parameters
        ----------
        docs : :class:`pymongo.cursor.Cursor`
            A query cursor with documents to include in the resampling.
        """
        if docs.count() > 0:
            self._mosaic_cursors.append(docs)

    def add_images_by_path(self, image_paths, weight_paths=None,
            noise_paths=None):
        """Rather than adding adding mosaics from a MosaicDB, directly add
        images from a list of paths.
        
        Parameters
        ----------
        image_paths : list
            List of FITS image paths
        weight_paths : list
            Optional list of FITS weight paths.
        noise_paths : list
            Optional list of FITS noise paths.
        """
        docs = []
        for i, image_path in enumerate(image_paths):
            doc = {'image_path': image_path,
                    '_id': os.path.splitext(os.path.basename(image_path))[0]}
            if weight_paths:
                doc['weight_path'] = weight_paths[i]
            if noise_paths:
                doc['noise_path'] = noise_paths[i]
            docs.append(doc)
        self._mosaic_cursors.append(docs)

    def resample(self, set_name, pix_scale=None, swarp_configs=None):
        """Resample mosaics to the given pixel scale.
        
        Mosaics will be identified in the MosaicDB with the ``set_name``.

        Parameters
        ----------
        set_name : str
            Name of the set of resampled images. This will be included in a
            field named ``'set_name'`` in the MosaicDB entries of the resampled
            documents. The mosaics will also have ``set_name`` appended to
            their ``_id`` names.
        pix_scale : float
            Pixel scale of the resampled mosaic, in arcseconds per
            pixel. Ensure it is compatible with any `target_fits` provided.
        swarp_configs : dict
            Optional configuration dictionary to pass to Swarp.

        Returns
        -------
        resamp_docs : list
            Returns a list of dictionaries describing each resampled image,
            if a MosaicDB was not supplied. Fields include `_id`, `image_path`,
            `weight_path`, and `noise_path`.
        """
        if swarp_configs:
            swarp_configs = dict(swarp_configs)
        swarp_configs['COMBINE'] = 'N'
        swarp_configs['RESAMPLE'] = 'Y'
        swarp_configs['PIXEL_SCALE'] = "%.2f" % pix_scale
        swarp_configs['PIXELSCALE_TYPE'] = 'MANUAL'
        swarp_configs['RESAMPLE_DIR'] = self.workdir
        swarp_configs['SUBTRACT_BACK'] = 'N'
        mosaic_docs = []
        mosaic_ids = []
        image_paths = []
        weight_paths = []
        noise_paths = []
        resamp_docs = []
        for cursor in self._mosaic_cursors:
            for doc in cursor:
                mosaic_docs.append(doc)
                mosaic_ids.append(doc['_id'])
                image_paths.append(doc['image_path'])
                if weight_paths is not None and 'weight_path' in doc:
                    weight_paths.append(doc['weight_path'])
                else:
                    weight_paths = None
                if noise_paths is not None and 'noise_path' in doc:
                    noise_paths.append(doc['noise_path'])
                else:
                    noise_paths = None
        resampled_image_paths, resampled_weight_paths = self._resample_images(
            set_name, swarp_configs,
            image_paths, weight_paths)
        if noise_paths:
            resampled_noise_paths = self._resample_noise(set_name,
                swarp_configs, resampled_image_paths, resampled_weight_paths,
                noise_paths)
        for i, (mosaic_id, base_doc, resamp_path) in enumerate(
                zip(mosaic_ids, mosaic_docs, resampled_image_paths)):
            mosaic_name = doc['_id'] + "_%s" % set_name
            doc = dict(base_doc)
            doc['_id'] = mosaic_name
            doc['source_image_path'] = base_doc['image_path']
            doc['image_path'] = resamp_path
            if weight_paths:
                doc['weight_path'] = resampled_weight_paths[i]
            if noise_paths:
                doc['noise_path'] = resampled_noise_paths[i]
            doc['set_name'] = set_name
            doc['pix_scale'] = pix_scale
            if 'couplings' in doc:
                del doc['couplings']
            if self.mosaicdb:
                self.mosaicdb.c.save(doc)
                self.mosaicdb.add_footprint_from_header(doc['_id'],
                    astropy.io.fits.getheader(resamp_path, 0))
            else:
                resamp_docs.append(doc)
        if resamp_docs:
            return resamp_docs

    def _resample_images(self, set_name, swarp_configs, image_paths,
            weight_paths):
        """Standard resampling of images."""
        rimage_paths = []
        rweight_paths = []

        # Resample first image and use it as a target frame for others
        if self._target_fits is None:
            swarp = moastro.astromatic.Swarp([image_paths[0]], set_name,
                weightPaths=[weight_paths[0]],
                configs=swarp_configs,
                workDir=self.workdir)
            swarp.run()
            resamp_paths, resamp_wpaths = swarp.resampled_paths([0])
            rimage_paths.append(resamp_paths[0]['0'])
            rweight_paths.append(resamp_wpaths[0]['0'])
            resample_indices = range(1, len(image_paths))
            self._target_fits = rimage_paths[0]
        else:
            resample_indices = range(len(image_paths))

        # Resample other images into same frame.
        _im_paths = [image_paths[i] for i in resample_indices]
        _w_paths = [weight_paths[i] for i in resample_indices]
        swarp = moastro.astromatic.Swarp(_im_paths, set_name,
            weightPaths=_w_paths,
            configs=swarp_configs,
            workDir=self.workdir)
        print "using target", self._target_fits
        swarp.set_target_fits(self._target_fits)
        swarp.run()
        resamp_paths, resamp_wpaths = swarp.resampled_paths([0])
        rimage_paths.extend([d['0'] for d in resamp_paths])
        rweight_paths.extend([d['0'] for d in resamp_paths])

        return rimage_paths, rweight_paths

    def _resample_noise(self, set_name, swarp_configs, image_paths,
            weight_paths, noise_paths):
        """Resampling of noise maps."""
        # First make variance images.
        var_paths = []
        resamp_noise_paths = []
        for noise_path in noise_paths:
            var_path = os.path.basename(os.path.splitext(noise_path)[0]) \
                + "_var.fits"
            f = astropy.io.fits.open(noise_path)
            f[0].data = f[0].data ** 2.
            f.write(var_path, clobber=True)
            f.close()
            var_paths.append(var_path)

        # Need to swarp individually since we need to match the resampled imgs
        for image_path, weight_path, var_path in zip(image_paths, weight_paths,
                var_paths):
            swarp = moastro.astromatic.Swarp([var_path], set_name,
                weightPaths=[weight_path],
                configs=swarp_configs,
                workDir=self.workdir)
            swarp.set_target_fits(image_path)
            swarp.run()
            resamp_paths, resamp_wpaths = swarp.resampled_paths([0])
            resamp_varsum_path = resamp_paths[0]['0']
            resamp_varsum_wpath = resamp_wpaths[0]['0']
            os.remove(resamp_varsum_wpath)
            # Convert variance to sigma map
            fits = astropy.io.fits.open(resamp_varsum_path)
            fits[0].data = np.sqrt(fits[0].data)
            rnoisepath = os.path.splitext(image_path)[0] + ".noise.fits"
            fits.writeto(rnoisepath)
            resamp_noise_paths.append(rnoisepath)

        for p in var_paths:
            os.remove(p)

        return resamp_noise_paths
