#!/usr/bin/env python
# encoding: utf-8
"""
Store footprints of FITS images, useful for computing image overlaps.

.. note:: this will eventually be deprecated with Mo'Astro image logs.
"""

import pymongo
import astropy.wcs
import numpy as np
from shapely.geometry import Polygon

from difftools import ManualResampledWCS


class FootprintDB(object):
    """Manages spatial footprints of images and mosaics."""
    def __init__(self, url=None, port=27017, dbname=None,
            cname="footprints"):
        super(FootprintDB, self).__init__()
        self.url = url
        self.port = port
        self.dbname = dbname
        self.cname = cname

        connection = pymongo.Connection(host=self.url, port=self.port)
        self.db = connection[self.dbname]
        self.c = self.db[self.cname]

    def new_from_header(self, header, **meta):
        """Create a footprint from the WCS embedded in a astropy.io.fits
        header.
        """
        wcs = astropy.wcs.WCS(header)
        self.new_from_wcs(wcs, **meta)

    def new_from_wcs(self, wcs, **meta):
        """Create a footprint from a astropy.wcs WCS instance.

        In a sense, this is an attempt to persist a WCS instance. Note that
        we only save a subset of the astropy.wcs data; that is, we're built
        around simple WCS with no PV, etc. This could be fixed though...
        
        .. note:: By default astropy.wcs is 1 based (ie, origins of CRPIX are
           1 and not zero; may need to subtract 1 from crpix when used in
           numpy arrays
        """
        naxis = (wcs.naxis1, wcs.naxis2)  # hopefully WCS has this from header
        crpix = tuple(wcs.wcs.crpix)  # (CRPIX1, CRPIX2)
        crval = tuple(wcs.wcs.crval)  # (CRVAL1, CRVAL2)
        ctype = tuple(wcs.wcs.ctype)
        # Make CD array, cast as list
        cd = []
        for (cdi, cdj) in wcs.wcs.cd:
            cd.append([cdi, cdj])
        # Make footprint polygon, cast to a list
        raDecFootprintArray = wcs.calcFootprint()
        raDecFootprint = []
        for (ra, dec) in raDecFootprintArray:
            raDecFootprint.append([ra, dec])

        doc = {"naxis": naxis, "ctype": ctype, "crpix": crpix, "crval": crval,
                "cd": cd, "radec_poly": raDecFootprint}
        doc.update(meta)

        self.c.remove(meta)  # remove identical footprints
        self.c.insert(doc)

    def new_from_hull_of_mef_images(self, imageLog, selector, nExt, **meta):
        """Create a field from the convex hull of the positions of a collection
        of images. Useful to find the approximate footprint from a set of
        images before making a formal mosaic.
        
        Parameters
        ----------
        imageLog : ImageLog
           an ImageLog instance
        selector : dict
           a valid MongoDB query for the image log
        nExt : int
           number of FITS extensions to expect in the images
        meta : dict
           fields to be added to the footprint db document
        """
        imageDocs = imageLog.getiter(selector, None)
        print imageDocs
        print "Found %i images with %s" % (imageDocs.count(), selector)
        raVertices = []
        decVertices = []
        for doc in imageDocs:
            print doc.keys()
            for ext in range(1, nExt + 1):
                extPoly = doc[str(ext)]['footprint']
                for (ra, dec) in extPoly:
                    raVertices.append(ra)
                    decVertices.append(dec)
        # now compute the convex hull of these points
        #allVertices = zip(raVertices, decVertices)
        #p = Polygon.Polygon(allVertices)
        #p = Polygon.Utils.fillHoles(p)
        #hull = Polygon.Utils.convexHull(p)
        #pointList = Polygon.Utils.pointList(hull)
        # Compute a naieve complex hull (box) from these points
        pointList = [[min(raVertices), min(decVertices)],
                [min(raVertices), max(decVertices)],
                [max(raVertices), max(decVertices)],
                [max(raVertices), min(decVertices)]]
        self.new_from_polygon(pointList, **meta)

    def new_from_polygon(self, poly, **meta):
        """Create a footprint from a polygon with (RA,Dec) vertices."""
        doc = {"radec_poly": poly}
        doc.update(meta)
        self.c.remove(meta)
        self.c.insert(doc)

    def find(self, selector, one=False):
        """Return the Footprint, or list of Footprints, requested by
        the query document `selector`.
        """
        if one:
            return self.c.find_one(selector)
        else:
            return self.c.find(selector)

    def find_overlapping(self, principleSel, selector):
        """Returns a list of footprint IDs for footprints overlapping that
        of the principle footprint, and the fractional area of the overlap
        compared to the area of the principle footprint.

        :param principleSel: selector of `FootprintDB` document to search
            against
        :param selector: `FootprintDB` query document to select fields to test
            overlaps with.
        :return: Sequence of `(_id, overlap fraction)`.
        """
        print "In find_overlapping"
        principleDoc = self.find(principleSel, one=True)
        print principleDoc
        verts = np.array(principleDoc['radec_poly'])
        print verts
        print np.mean(verts, axis=0)
        ra0, dec0 = np.mean(verts, axis=0)
        xi, eta = eq_to_tan(verts[:, 0], verts[:, 1], ra0, dec0)
        print xi, eta
        principlePoly = Polygon(zip(xi, eta))
        # TODO need to implement an RA, Dec centroid and perform spatial
        # queries against those as a first pass
        overlaps = []
        for doc in self.find(selector):
            fieldVerts = np.array(doc['radec_poly'])
            xi, eta = eq_to_tan(fieldVerts[:, 0], fieldVerts[:, 1], ra0, dec0)
            poly = Polygon(zip(xi, eta))
            if principlePoly.intersects(poly):
                print "overlap!", doc['field']
                interArea = principlePoly.intersection(poly).area
                fracOverlap = interArea / principlePoly.area
                overlaps.append((doc['_id'], fracOverlap))
        return overlaps

    def make_resampled_wcs(self, selector):
        """Make a list (or one) ResampledWCS object(s) for footprints
        given by the selector"""
        docs = self.find(selector)
        print "make_resampled_wcs"
        print "-- selector:", selector
        print "-- count:", docs.count()
        wcsList = []
        for doc in docs:
            naxis1, naxis2 = doc['naxis']
            crpix1, crpix2 = doc['crpix']
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
