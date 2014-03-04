#!/usr/bin/env python
# encoding: utf-8
"""
Apply polygon flag regions and images to blank out weightmaps and generate flag
regions.
"""
import subprocess
import astropy.io.fits


def make_flagmap(image_path, output_flag_path, output_weight_path=None,
        weight_path=None, flag_path=None, region_paths=None,
        min_weight=0.3, max_weight=None):
    """Generate a flagmap of bad pixel values, where 1=bad, 0=good. Merges
    information from an existing flagmap, a weightmap, and from polygons.

    Parameters
    ----------

    image_path : str
        Path to the original image.
    output_flag_path : str
        Path to the output flagmap.
    output_weight_path : str
        Optional path to output weightmap, where bad pixels are set to zero.
    weight_path : str
        Optional path to input weightmap (if  used to filter bad pixels)
    flag_path : str
        Optinal path to existing flagmap.
    region_paths : list or str
        Optional list to existing DS9 region files, where pixels inside
        regions will be masked.
    """
    ww = WeightWatcher()
    ww.set_param("OUTFLAG_NAME", output_flag_path)
    if output_weight_path:
        ww.set_param("OUTWEIGHT_NAME", output_weight_path)
    if weight_path:
        ww.set_param("WEIGHT_NAMES", weight_path)
        if min_weight:
            ww.set_param("WEIGHT_MIN", min_weight)
        if max_weight:
            ww.set_param("WEIGHT_MAX", max_weight)
    if flag_path:
        ww.set_param("FLAG_NAMES", flag_path)
        # TODO set FLAG_WMASKS and FLAG_OUTFLAGS/FLAG_MASKS
    if region_paths:
        if isinstance(region_paths, str):
            region_paths = [region_paths]
        # TODO verify regions are polygons in image coordinates
        ww.set_param("POLY_NAMES", region_paths)
        ww.set_param("POLY_OUTFLAGS", 1.)
        ww.set_param("POLY_OUTWEIGHT", 0.)
    ww.run()
    return ww.output_flag_path, ww.output_weight_path


def convert_region(image_path, region_path, workdir):
    """Convert regions in the DS9 file to polygons in image coordinates."""
    import pyregion
    # TODO needs to be implemented; doing conversions manually now
    reg = pyregion.open(region_path)
    reg.as_imagecoord(astropy.io.fits.getheader(image_path, 0))


class WeightWatcher(object):
    """Wrapper to WeightWatcher from astromatic."""
    def __init__(self):
        super(WeightWatcher, self).__init__()
        # Initialize parameters with null strings for paths
        self._params = {"FLAG_NAMES": '""', "WEIGHT_NAMES": '""',
                "OUTWEIGHT_NAME": '""', "OUTFLAG_NAME": '""'}

    def set_param(self, key, val):
        """Sets the value of a weightwatcher parameter."""
        if isinstance(val, list) or isinstance(val, tuple):
            val = ",".join(str(v) for v in val)
        self._params[key] = str(val)

    def run(self):
        """Run weight watcher"""
        args = " ".join(["-%s %s" % (k, v)
            for k, v in self._params.iteritems()])
        cmd = "ww %s" % args
        print cmd
        subprocess.call(cmd, shell=True)

    @property
    def output_flag_path(self):
        """Path to the output flag map."""
        return self._output_path("OUTFLAG_NAME")

    @property
    def output_weight_path(self):
        """Path to the output weight map."""
        return self._output_path("OUTWEIGHT_NAME")

    def _output_path(self, key):
        if self._params[key] == '""':
            return None
        else:
            return self._params[key]
