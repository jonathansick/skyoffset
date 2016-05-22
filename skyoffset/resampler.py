#!/usr/bin/env python
# encoding: utf-8
"""
The resampler module provides tools for resampling a set of mosaics, both
to have matching pixel grids, and to alter the pixel scale while propagating
noise maps.
"""

import os
import shutil

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
        self.MAX_NPIX_IN_MEMORY = 0.4e9  # max n pixel before using mmap
        self.MEMMAP_CHUNK_SIZE = 1000  # number of rows to process in memmap

        self.mosaicdb = mosaicdb

        self._mosaic_docs = []
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
        for doc in docs:
            self._mosaic_docs.append(doc)

    def add_images_by_path(self, image_paths, weight_paths=None,
                           noise_paths=None, flag_paths=None,
                           offset_zp_sigmas=None):
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
        flag_paths : list
            Optional list of FITS flag paths, where 0 is a permitted pixel,
            and pixels > 1 are set to NaN prior to resampling.
        offset_zp_sigmas : list
            Optional list of sky background uncertainties to be propagated
            into the document keyword `'offset_zp_sigma'` (in image pixel
            units).
        """
        for i, image_path in enumerate(image_paths):
            doc = {'image_path': image_path,
                   '_id': os.path.splitext(os.path.basename(image_path))[0]}
            if weight_paths:
                doc['weight_path'] = weight_paths[i]
            if noise_paths:
                doc['noise_path'] = noise_paths[i]
            if flag_paths:
                doc['flag_path'] = flag_paths[i]
            if offset_zp_sigmas:
                doc['offset_zp_sigma'] = offset_zp_sigmas[i]
            header = astropy.io.fits.getheader(image_path, 0)
            pix_scale = np.sqrt(header['CD1_1'] ** 2.
                                + header['CD1_2'] ** 2.) * 3600.
            doc['pix_scale'] = pix_scale
            self._mosaic_docs.append(doc)

    def resample(self, set_name, pix_scale=None, swarp_configs=None,
                 path_key='image_path'):
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
        else:
            swarp_configs = {}
        swarp_configs['COMBINE'] = 'N'
        swarp_configs['RESAMPLE'] = 'Y'
        swarp_configs['PIXEL_SCALE'] = "%.2f" % pix_scale
        swarp_configs['PIXELSCALE_TYPE'] = 'MANUAL'
        swarp_configs['RESAMPLE_DIR'] = self.workdir
        swarp_configs['SUBTRACT_BACK'] = 'N'

        # Make lists of input images
        mosaic_docs = []
        mosaic_ids = []
        image_paths = []
        weight_paths = []
        noise_paths = []
        resamp_docs = []
        temp_image_paths = []  # for flagged images
        for doc in self._mosaic_docs:
            mosaic_docs.append(doc)
            mosaic_ids.append(doc['_id'])
            if 'flag_path' in doc:
                # set flagged pixels to nan before resampling
                tmp_im_path = os.path.join(
                    self.workdir,
                    doc['_id'] + ".flagged.fits")
                self._set_flagged_nans(
                    doc[path_key], doc['flag_path'],
                    tmp_im_path)
                temp_image_paths.append(tmp_im_path)
                image_paths.append(tmp_im_path)
            else:
                image_paths.append(doc[path_key])
            if weight_paths is not None and 'weight_path' in doc:
                weight_paths.append(doc['weight_path'])
            else:
                weight_paths = None
            if 'noise_path' in doc:
                noise_paths.append(doc['noise_path'])
            else:
                noise_paths.append(None)

        # Resample images and weight paths
        resampled_image_paths, resampled_weight_paths = self._resample_images(
            set_name, swarp_configs,
            image_paths, weight_paths)

        for i, (mosaic_id, base_doc, resamp_path) in enumerate(
                zip(mosaic_ids, mosaic_docs, resampled_image_paths)):
            mosaic_name = doc['_id'] + "_%s" % set_name
            doc = dict(base_doc)
            orig_pix_scale = base_doc['pix_scale']
            doc['_id'] = mosaic_name
            doc['source_image_path'] = base_doc[path_key]
            doc['image_path'] = resamp_path
            if weight_paths is not None:
                doc['weight_path'] = resampled_weight_paths[i]
            doc['set_name'] = set_name
            doc['pix_scale'] = pix_scale
            doc['native_pix_scale'] = orig_pix_scale
            if 'offsets' in doc:
                doc['offsets'] = self._rescale_offsets(
                    doc['offsets'],
                    orig_pix_scale, pix_scale)
            if 'offset_zp_sigma' in doc:
                doc['offset_zp_sigma'] = self._rescale_offset_zp_sigma(
                    doc['offset_zp_sigma'], orig_pix_scale, pix_scale)
            if 'couplings' in doc:
                del doc['couplings']
            # Resample the noise frame now, if it exists
            if noise_paths[i] is not None:
                resamp_noise_path = self._resample_noise(
                    set_name,
                    swarp_configs, [resamp_path],
                    [resampled_weight_paths[i]],
                    [noise_paths[i]])[0]
                doc['noise_path'] = resamp_noise_path
            if self.mosaicdb:
                self.mosaicdb.c.save(doc)
                self.mosaicdb.add_footprint_from_header(
                    doc['_id'],
                    astropy.io.fits.getheader(resamp_path, 0))
            else:
                resamp_docs.append(doc)

        for path in temp_image_paths:
            if os.path.exists(path):
                os.remove(path)

        if resamp_docs:
            return resamp_docs

    def _set_flagged_nans(self, image_path, flag_path, output_path):
        """Create a version of a FITS image where flagged pixel are set to NaN.
        """
        assert output_path != image_path
        nx = astropy.io.fits.getval(image_path, 'NAXIS1', 0)
        ny = astropy.io.fits.getval(image_path, 'NAXIS2', 0)
        npix = nx * ny
        if npix > self.MAX_NPIX_IN_MEMORY:
            print "Using mmemap to make flagged image", output_path
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.copy(image_path, output_path)
            ffits = astropy.io.fits.open(flag_path, memmap=True)
            ofits = astropy.io.fits.open(output_path, memmap=True,
                                         mode='update')
            ymin = 0
            ymax = ymin + self.MEMMAP_CHUNK_SIZE
            while ymax <= ny:
                ofits[0].data[ymin:ymax, :][ffits[0].data[ymin:ymax, :] > 0] \
                    = np.nan
                ofits.flush()
                # Update chunck range
                ymin = ymax
                ymax = ymin + self.MEMMAP_CHUNK_SIZE
                if ymax > ny:
                    ymax = ny
                if ymin == ny:
                    break
            ffits.close()
            ofits.close()
        else:
            f = astropy.io.fits.open(image_path)
            flagfits = astropy.io.fits.open(flag_path)
            f[0].data[flagfits[0].data > 0] = np.nan
            f.writeto(output_path, clobber=True)
            f.close()
            flagfits.close()

    def _resample_images(self, set_name, swarp_configs, image_paths,
                         weight_paths):
        """Standard resampling of images."""
        rimage_paths = []
        rweight_paths = []

        # Resample first image and use it as a target frame for others
        if self._target_fits is None:
            if weight_paths is not None:
                target_weight = [weight_paths[0]]
            else:
                target_weight = None
            swarp = moastro.astromatic.Swarp(
                [image_paths[0]], set_name,
                weightPaths=target_weight,
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
        if weight_paths is not None:
            _w_paths = [weight_paths[i] for i in resample_indices]
        else:
            _w_paths = None
        if len(_im_paths) > 0:
            swarp = moastro.astromatic.Swarp(
                _im_paths, set_name,
                weightPaths=_w_paths,
                configs=swarp_configs,
                workDir=self.workdir)
            swarp.set_target_fits(self._target_fits)
            swarp.run()
            resamp_paths, resamp_wpaths = swarp.resampled_paths([0])
            # Trim the resampled images to ensure NAXISi and CRPIXi match
            rimage_paths.extend([d['0'] for d in resamp_paths])
            rweight_paths.extend([d['0'] for d in resamp_wpaths])
            self._resize_resampled_images(self._target_fits, rimage_paths)
            self._resize_resampled_images(self._target_fits, rweight_paths)

        return rimage_paths, rweight_paths

    def _resize_resampled_images(self, target_fits_path, resampled_paths):
        """Ensures that the CRPIX, CRVAL and NAXIS1/2 values of the resampled
        images match the target, and crops/adds padding if not. Swarp *should*
        do this properly, but sometimes does not.
        """
        target_fits = astropy.io.fits.open(target_fits_path)
        rNAXIS1 = target_fits[0].header['NAXIS1']
        rNAXIS2 = target_fits[0].header['NAXIS2']
        rCRPIX1 = target_fits[0].header['CRPIX1']
        rCRPIX2 = target_fits[0].header['CRPIX2']
        for path in resampled_paths:
            touched = False  # toggled True if image modified
            print "path", path
            fits = astropy.io.fits.open(path)
            image = fits[0].data
            print "orig shape", image.shape

            # x-axis
            if rCRPIX1 > fits[0].header['CRPIX1']:
                # pad from left
                print "CRPIX1 conflict %i %i" \
                    % (rCRPIX1, fits[0].header['CRPIX1'])
                dx = rCRPIX1 - fits[0].header['CRPIX1']
                print "Pad left by %i" % dx
                pad = np.ones((image.shape[0], dx)) * np.nan
                image = np.hstack((pad, image))
                print image.shape
                touched = True
            elif rCRPIX1 < fits[0].header['CRPIX1']:
                # trim from left
                print "CRPIX1 conflict %i %i" \
                    % (rCRPIX1, fits[0].header['CRPIX1'])
                dx = fits[0].header['CRPIX1'] - rCRPIX1
                print "Trim left by %i" % dx
                image = image[:, dx:]
                print image.shape
                touched = True
            if rNAXIS1 > image.shape[1]:
                # pad to the right
                print "NAXIS1 conflict %i %i" % (rNAXIS1, image.shape[1])
                dx = rNAXIS1 - image.shape[1]
                print "Pad from right by %i" % dx
                pad = np.ones((image.shape[0], dx)) * np.nan
                image = np.hstack((image, pad))
                print image.shape
                touched = True
            elif rNAXIS1 < image.shape[1]:
                # trim from right
                print "NAXIS1 conflict %i %i" % (rNAXIS1, image.shape[1])
                dx = image.shape[1] - rNAXIS1
                print "Trim from right by %i" % dx
                image = image[:, :-dx]
                print image.shape
                touched = True

            # y-axis
            crpix2 = fits[0].header['CRPIX2']
            if rCRPIX2 > crpix2:
                # Pad from bottom (low index in image array)
                print "pad from bottom"
                dx = rCRPIX2 - crpix2
                pad = np.ones((dx, image.shape[1])) * np.nan
                image = np.vstack((pad, image))
                touched = True
            elif rCRPIX2 < crpix2:
                # Trim from bottom (low index in image array)
                print "trim from bottom"
                dx = crpix2 - rCRPIX2
                image = image[dx:, :]
                touched = True
            if rNAXIS2 > image.shape[0]:
                # Pad from top (high index in image array)
                print "pad from top"
                dx = rNAXIS2 - image.shape[0]
                pad = np.ones((dx, image.shape[1])) * np.nan
                image = np.vstack((image, pad))
                touched = True
            elif rNAXIS2 < image.shape[0]:
                # Trim from top (high index in image array)
                print "trim from top"
                dx = rNAXIS2 - image.shape[0]
                image = image[:-dx, :]
                touched = True

            if touched:
                fits[0].data = image
                fits[0].header.update('NAXIS1', image.shape[1])
                fits[0].header.update('NAXIS2', image.shape[0])
                fits[0].header.update('CRPIX1', rCRPIX1)
                fits[0].header.update('CRPIX2', rCRPIX2)
                fits.writeto(path, clobber=True)
            fits.close()

        target_fits.close()

    def _rescale_offsets(self, offsets, orig_pixscale, new_pixscale):
        """Scale the sky offset dictionary according to astrometric scaling."""
        scale_factor = new_pixscale ** 2. / orig_pixscale ** 2.
        scaled_offsets = {k: v * scale_factor for k, v in offsets.iteritems()}
        return scaled_offsets

    def _rescale_offset_zp_sigma(self, offset_zp_scale, orig_pixscale,
                                 new_pixscale):
        """Scale the sky offset zp uncertainty according to astrometric
        scaling.
        """
        scale_factor = new_pixscale ** 2. / orig_pixscale ** 2.
        return offset_zp_scale * scale_factor

    def _resample_noise(self, set_name, swarp_configs, image_paths,
                        weight_paths, noise_paths):
        """Resampling of noise maps."""
        # First make variance images.
        var_paths = []
        resamp_noise_paths = []
        for noise_path in noise_paths:
            var_path = os.path.basename(os.path.splitext(noise_path)[0]) \
                + "_var.fits"
            self._make_variance_map(noise_path, var_path)
            var_paths.append(var_path)

        # Need to swarp individually since we need to match the resampled imgs
        for image_path, weight_path, var_path in zip(image_paths, weight_paths,
                                                     var_paths):
            print "var_path", var_path
            print "weight_path", weight_path
            swarp = moastro.astromatic.Swarp(
                [var_path], set_name,
                # weightPaths=[weight_path],
                configs=swarp_configs,
                workDir=self.workdir)
            swarp.set_target_fits(image_path)
            swarp.run()
            resamp_paths, resamp_wpaths = swarp.resampled_paths([0])
            resamp_varsum_path = resamp_paths[0]['0']
            resamp_varsum_wpath = resamp_wpaths[0]['0']
            print "resamp_varsum_path", resamp_varsum_path
            print "resamp_varsum_wpath", resamp_varsum_wpath
            os.remove(resamp_varsum_wpath)
            # Convert variance to sigma map
            rnoisepath = os.path.splitext(image_path)[0] + ".noise.fits"
            self._make_sigma_map(resamp_varsum_path, rnoisepath)
            resamp_noise_paths.append(rnoisepath)

        for p in var_paths:
            os.remove(p)

        self._resize_resampled_images(self._target_fits, resamp_noise_paths)
        return resamp_noise_paths

    def _make_variance_map(self, sigma_path, var_path):
        nx = astropy.io.fits.getval(sigma_path, 'NAXIS1', 0)
        ny = astropy.io.fits.getval(sigma_path, 'NAXIS2', 0)
        npix = nx * ny
        if npix > self.MAX_NPIX_IN_MEMORY:
            print "Using memmap for variance map", var_path
            if os.path.exists(var_path):
                os.remove(var_path)
            shutil.copy(sigma_path, var_path)
            sigma_fits = astropy.io.fits.open(sigma_path, memmap=True)
            var_fits = astropy.io.fits.open(var_path, memmap=True,
                                            mode='update')
            ymin = 0
            ymax = ymin + self.MEMMAP_CHUNK_SIZE
            while ymax <= ny:
                var_fits[0].data[ymin:ymax, :] \
                    = sigma_fits[0].data[ymin:ymax, :] ** 2.
                var_fits.flush()
                # Update chunck range
                ymin = ymax
                ymax = ymin + self.MEMMAP_CHUNK_SIZE
                if ymax > ny:
                    ymax = ny
                if ymin == ny:
                    break
            sigma_fits.close()
            var_fits.close()
        else:
            f = astropy.io.fits.open(sigma_path)
            f[0].data = f[0].data ** 2.
            f.writeto(var_path, clobber=True)
            f.close()

    def _make_sigma_map(self, var_path, sigma_path):
        nx = astropy.io.fits.getval(var_path, 'NAXIS1', 0)
        ny = astropy.io.fits.getval(var_path, 'NAXIS2', 0)
        npix = nx * ny
        if npix > self.MAX_NPIX_IN_MEMORY:
            print "Using memmap for sigma map", sigma_path
            if os.path.exists(sigma_path):
                os.remove(sigma_path)
            shutil.copy(var_path, sigma_path)
            var_fits = astropy.io.fits.open(var_path, memmap=True)
            sigma_fits = astropy.io.fits.open(sigma_path, memmap=True,
                                              mode='update')
            ymin = 0
            ymax = ymin + self.MEMMAP_CHUNK_SIZE
            while ymax <= ny:
                sigma_fits[0].data[ymin:ymax, :] = np.sqrt(
                    var_fits[0].data[ymin:ymax, :])
                sigma_fits.flush()
                # Update chunck range
                ymin = ymax
                ymax = ymin + self.MEMMAP_CHUNK_SIZE
                if ymax > ny:
                    ymax = ny
                if ymin == ny:
                    break
            sigma_fits.close()
            var_fits.close()
        else:
            f = astropy.io.fits.open(var_path)
            f[0].data = np.sqrt(f[0].data)
            f.writeto(sigma_path, clobber=True)
            f.close()
