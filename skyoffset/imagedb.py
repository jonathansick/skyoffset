#!/usr/bin/env python
# encoding: utf-8
"""
MongoDB databases for storing references to mosaic image products (at the
stack, block and mosaic level).

The databases are backed by the Mo'Astro ImageLog API. Most of the additional
methods provided for each database are to preserve backwards compatibility.
These subclasses provide additional functionality for storing footprints
within a common resampled pixel space.
"""

import numpy as np
import astropy.io.fits
import astropy.wcs
from shapely.geometry import Polygon

from moastro.imagelog import ImageLog

from difftools import ManualResampledWCS


class MosaicDB(ImageLog):
    """Database interface for resampled mosaics.
    
    Parameters
    ----------
    dbname : str
        Name of the MongoDB database mosaic documents are stored in.
    cname : str
        Name of the MongoDB collection where mosaic docs are stored.
    server : str
        Name of the MongoDB server (using a `.moastro.json` config).
    url : str
        Hostname/URL of the MongoDB server (if ``server`` is not set).
    port : int
        Port of the MongoDB server (if ``server`` is not set).
    """
    def __init__(self, dbname, cname,
            server=None, url="localhost", port=27017):
        super(MosaicDB, self).__init__(dbname, cname,
                server=server, url=url, port=port)

    def add_footprint_from_header(self, mosaic_name, header):
        """Create a footprint from the WCS embedded in a ``astropy.io.fits``
        header for the named (pre-existing) mosaic document.

        The footprint is stored under the field ``footprint`` in the mosaic's
        document.

        Parameters
        ----------
        mosaic_name : str
            Name of the mosaic (the ``_id`` field for the document).
        header : :class:`astropy.io.fits.Header` instance
            Header for the mosaic.
        """
        wcs = astropy.wcs.WCS(header)
        self.add_footprint_from_wcs(mosaic_name, wcs)

    def add_footprint_from_wcs(self, mosaic_name, wcs):
        """Create a footprint from the WCS embedded in a ``astropy.io.fits``
        header for the named (pre-existing) mosaic document.

        The footprint is stored under the field ``footprint`` in the mosaic's
        document.

        In a sense, this is an attempt to persist a WCS instance. Note that
        we only save a subset of the astropy.wcs data; that is, we're built
        around simple WCS with no PV, etc. This could be fixed though...
        
        .. note:: By default astropy.wcs is 1 based (ie, origins of CRPIX are
           1 and not zero; may need to subtract 1 from crpix when used in
           numpy arrays

        Parameters
        ----------
        mosaic_name : str
            Name of the mosaic (the ``_id`` field for the document).
        wcs : :class:`astropy.wcs.WCS` instance
            WCS for the mosaic.
        """
        doc = {}
        doc['naxis'] = (wcs._naxis1, wcs._naxis2)
        doc['crpix'] = tuple(wcs.wcs.crpix)  # (CRPIX1, CRPIX2)
        doc['crval'] = tuple(wcs.wcs.crval)  # (CRVAL1, CRVAL2)
        doc['ctype'] = tuple(wcs.wcs.ctype)
        if wcs.wcs.has_cd():
            cd = []
            for (cdi, cdj) in wcs.wcs.cd:
                cd.append([cdi, cdj])
            doc['cd'] = cd
        try:
            doc['cdelt'] = tuple(wcs.wcs.cdelt)
        except:
            pass
        try:
            doc['crota'] = tuple(wcs.wcs.crota)
        except:
            pass
        # Make footprint polygon, cast to a list
        raDecFootprintArray = wcs.calcFootprint()
        raDecFootprint = []
        for (ra, dec) in raDecFootprintArray:
            raDecFootprint.append([ra, dec])
        doc['radec_poly'] = raDecFootprint
        # Add footprint to mosaic document
        self.set(mosaic_name, "footprint", doc)

    def find_overlapping(self, mosaic_name, selector):
        """Returns a list of mosaic names (``_id``) for mosaics overlapping
        the principal mosaic, and the fractional area of the overlap
        compared to the area of the principal footprint.

        Parameters
        ----------
        mosaic_name : str
            `_id` of the mosaic to test other mosaics against.
        selector : dict
            query document to select mosaics to test
            overlaps with.

        Returns
        -------
        overlaps : list
            Sequence of `(_id, overlap fraction)` tuples.
        """
        main_doc = self.find({"_id": mosaic_name}, one=True)
        verts = np.array(main_doc['footprint.radec_poly'])
        ra0, dec0 = np.mean(verts, axis=0)
        xi, eta = eq_to_tan(verts[:, 0], verts[:, 1], ra0, dec0)
        main_poly = Polygon(zip(xi, eta))
        # need to implement an RA, Dec centroid and perform spatial
        # queries against those as a first pass
        overlaps = []
        for doc in self.find(selector):
            field_verts = np.array(doc['footprint.radec_poly'])
            xi, eta = eq_to_tan(field_verts[:, 0], field_verts[:, 1],
                    ra0, dec0)
            poly = Polygon(zip(xi, eta))
            if main_poly.intersects(poly):
                iter_area = main_poly.intersection(poly).area
                frac_overlap = iter_area / main_poly.area
                overlaps.append((doc['_id'], frac_overlap))
        return overlaps

    def make_resampled_wcs(self, selector):
        """Make a list (or one) ResampledWCS object(s) for footprints
        given by the selector.
        
        Parameters
        ----------
        selector : dict
            MongoDB selector for mosaic documents to make ``ResampledWCS``
            instances for.

        Returns
        -------
        resampled_wcs : list or :class:`skyoffset.difftools.ResampledWCS`
            If only one mosaic is found, returns a single
            :class:`skyoffset.difftools.ResampledWCS` instance. Otherwise,
            a list of ``ResampledWCS``s is returned. ``None`` is returned
            if no mosaics match the selector.
        """
        docs = self.find(selector)
        wcsList = []
        for doc in docs:
            naxis1, naxis2 = doc['footprint.naxis']
            crpix1, crpix2 = doc['footprint.crpix']
            resampledWCS = ManualResampledWCS(naxis1, naxis2, crpix1, crpix2)
            wcsList.append(resampledWCS)

        if len(wcsList) > 1:
            return wcsList
        elif len(wcsList) > 0:
            return wcsList[0]
        else:
            return None


def eq_to_tan(ra, dec, ra0, dec0):
    """Converts RA,Dec coordinates to xi, eta tangential coordiantes.
    See Olkin:1996 eq 3 for example, or Smart 1977.

    :return: tuple of xi, eta in degrees.
    """
    r = ra * np.pi / 180.
    d = dec * np.pi / 180.
    r0 = ra0 * np.pi / 180.
    d0 = dec0 * np.pi / 180.

    xi = np.cos(d) * np.sin(r - r0) \
        / (np.sin(d0) * np.sin(d)
        + np.cos(d0) * np.cos(d) * np.cos(r - r0))

    eta = (np.cos(d0) * np.sin(d)
        - np.sin(d0) * np.cos(d) * np.cos(r - r0)) \
        / (np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(r - r0))

    xi = xi * 180. / np.pi
    eta = eta * 180. / np.pi
    return xi, eta


def tan_to_eq(xiDeg, etaDeg, ra0Deg, dec0Deg):
    """Convert tangential coordinates to equatorial (RA, Dec) in degrees."""
    xi = xiDeg * np.pi / 180.
    eta = etaDeg * np.pi / 180.
    ra0 = ra0Deg * np.pi / 180.
    dec0 = dec0Deg * np.pi / 180.

    ra = np.arctan(xi / (np.cos(dec0) - eta * np.sin(dec0))) + ra0
    dec = np.arctan((np.sin(dec0) + eta * np.cos(dec0))
            / (np.cos(dec0) - eta * np.sin(dec0))) * np.cos(ra - ra0)

    ra = ra * 180. / np.pi
    dec = dec * 180. / np.pi
    return ra, dec
