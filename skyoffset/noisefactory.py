#!/usr/bin/env python
# encoding: utf-8
"""
Module for creating co-added noise maps (as Gaussian sigma in image DN units).
"""

import os
import shutil

import numpy as np
import astropy.io.fits
from moastro.astromatic import Swarp


class NoiseMapFactory(object):
    """Construct the Gaussian noise map of a mosaic given the component images.
    
    Parameters
    ----------
    noise_paths : list
        List of paths to individual noise (Gaussian sigma) images in the
        mosaic's stack. These should be single extension FITS with an
        appropriate WCS and resampled into the mosaics coordinate system.
    weight_paths : list
        List of paths to individual resampled weight maps of images in the
        stack. Pixels with zero or negative weights are effectively
        masked from the coverage maps.
    mosaic_path : str
        Path to the mosaic image. This is needed to establish the target
        WCS frame of the mosaic.
    swarp_configs : dict
        Dictionary of configurations, to be passed to
        :class:`moastro.astromatic.Swarp`.
    delete_temps : bool
        Set ``False`` if intermediate image files *should not* be deleted.
    """
    def __init__(self, noise_paths, weight_paths, mosaic_path,
            swarp_configs=None, delete_temps=True):
        super(NoiseMapFactory, self).__init__()
        self.MAX_NPIX_IN_MEMORY = 0.4e9  # max n pixel before using mmap
        self.MEMMAP_CHUNK_SIZE = 1000  # number of rows to process in memmap
        self._paths = noise_paths
        self._weight_paths = weight_paths
        self._mosaic_path = mosaic_path
        self._noise_map_path = ".".join((os.path.splitext(mosaic_path)[0],
            "noise.fits"))
        if swarp_configs:
            self._configs = {}
        else:
            self._configs = dict(swarp_configs)
        # Swarp component images of the mosaic
        coverage_paths, variance_paths = self._make_temp_images()
        coverage_map_path = self._make_coverage_map(coverage_paths)
        var_sum_path = self._make_var_sum_map(variance_paths, coverage_paths)
        # Combine the maps into a sigma map
        self._make_sigma_map(coverage_map_path,
                var_sum_path)
        # Delete temporary images
        if delete_temps:
            for p in coverage_paths:
                os.remove(p)
            for p in variance_paths:
                os.remove(p)
            os.remove(var_sum_path)
            os.remove(coverage_map_path)

    @property
    def map_path(self):
        return self._noise_map_path

    def _make_temp_images(self):
        """Make pixel coverage and variance images."""
        cov_paths, var_paths = [], []
        for sigma_path, weight_path in zip(self._paths, self._weight_paths):
            coverage_path = ".".join((os.path.splitext(sigma_path)[0],
                "coverage.fits"))
            var_path = ".".join((os.path.splitext(sigma_path)[0],
                "var.fits"))
            fits = astropy.io.fits.open(sigma_path)
            wfits = astropy.io.fits.open(weight_path)
            # Make a coverage image from the weightmap
            wfits[0].data[wfits[0].data > 0.] = 1.
            wfits[0].data[wfits[0].data <= 0.] = 0.
            wfits.writeto(coverage_path, clobber=True)
            # Make a variance map
            fits[0].data = fits[0].data ** 2.
            fits[0].data[wfits[0].data == 0.] = 0.  # FIXME NaNs propagate bad?
            fits.writeto(var_path, clobber=True)
            fits.close()
            wfits.close()
            cov_paths.append(coverage_path)
            var_paths.append(var_path)
        return cov_paths, var_paths

    def _make_coverage_map(self, coverage_paths):
        """Make a map with the sum of images covering each pixel."""
        configs = dict(self._configs)
        configs.update({"COMBINE_TYPE": "SUM", "RESAMPLE": "N",
            "SUBTRACT_BACK": "N"})
        name = os.path.basename(os.path.splitext(self._mosaic_path)[0]) \
                + ".coverage.fits"
        swarp = Swarp(coverage_paths, name,
                configs=configs,
                workDir=os.path.dirname(self._mosaic_path))
        swarp.set_target_fits(self._mosaic_path)
        swarp.run()
        coadd_path, coadd_weight_path = swarp.mosaic_paths()
        os.remove(coadd_weight_path)
        return coadd_path

    def _make_var_sum_map(self, variance_paths, coverage_paths):
        """Make a map with the sum of variances."""
        configs = dict(self._configs)
        configs.update({"COMBINE_TYPE": "SUM", "RESAMPLE": "N",
            "WEIGHT_TYPE": "MAP_WEIGHT",
            "SUBTRACT_BACK": "N"})
        name = os.path.basename(os.path.splitext(self._mosaic_path)[0]) \
                + ".varsum.fits"
        swarp = Swarp(variance_paths, name,
                weightPaths=coverage_paths,
                configs=configs,
                workDir=os.path.dirname(self._mosaic_path))
        swarp.set_target_fits(self._mosaic_path)
        swarp.run()
        coadd_path, coadd_weight_path = swarp.mosaic_paths()
        os.remove(coadd_weight_path)
        return coadd_path

    def _make_sigma_map(self, coverage_path, variance_sum_path):
        """Make a final sigma map.
        
        Since mosaics can be potentially very large, we open the coverage
        and variance sum maps using a memory map and operate on chunks of rows
        if the mosaic is larger than a defined size.
        """
        nx = astropy.io.fits.getval(variance_sum_path, 'NAXIS1', 0)
        ny = astropy.io.fits.getval(variance_sum_path, 'NAXIS2', 0)
        npix = nx * ny
        if npix > self.MAX_NPIX_IN_MEMORY:
            print "Using memmap for sigma map"
            # mosaic is large enough to warrant memory mapping
            # Process the noise map line-by-line
            cfits = astropy.io.fits.open(coverage_path, memmap=True)
            shutil.copy(variance_sum_path, self._noise_map_path)
            vfits = astropy.io.fits.open(variance_sum_path,
                    mode='update', memmap=True)
            ymin = 0
            ymax = ymin + self.MEMMAP_CHUNK_SIZE
            while ymax <= ny:
                vfits[0].data[ymin:ymax, :] = np.sqrt(
                    vfits[0].data[ymin:ymax, :]
                    / cfits[0].data[ymin:ymax, :])
                vfits.flush()
                # Update chunck range
                ymin = ymax
                ymax = ymin + self.MEMMAP_CHUNK_SIZE
                if ymax > ny:
                    ymax = ny
                if ymin == ny:
                    break
            cfits.close()
            vfits.close()
        else:
            print "In-memory for sigma map"
            # Operate on entire mosaic in memory
            cfits = astropy.io.fits.open(coverage_path)
            vfits = astropy.io.fits.open(variance_sum_path)
            vfits[0].data = np.sqrt(vfits[0].data / cfits[0].data)
            vfits.writeto(self._noise_map_path, clobber=True)
            cfits.close()
            vfits.close()
