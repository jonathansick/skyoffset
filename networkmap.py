#!/usr/bin/env python
# encoding: utf-8
"""
For plotting network connections between blocks.

History
-------
2011-09-18 - Created by Jonathan Sick
"""

__all__ = ['']

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from owl.Footprint import equatorialToTan
from andpipe.footprintdb import FootprintDB
from mosaicdb import MosaicDB
from residualfluxes import ScalarResidualFlux

M31RA0 = 10.6846833
M31DEC0 = 41.2690361

def main():
    plot_scalar_relative_residual("skysubpub/scalar_relresidnet")
    plot_scalar_residual_sb("skysubpub/scalar_residual_sb_net")

def plot_scalar_relative_residual(plotPath):
    """docstring for compute_residual_scalar_flux"""
    fig = plt.figure(figsize=(6,4.2), frameon=False)
    fig.subplots_adjust(left=0.02, bottom=0.0, right=0.98, top=0.99,
            wspace=0.1, hspace=None)
    axes = [fig.add_subplot(121, frame_on=False),
            fig.add_subplot(122, frame_on=False)]
    bands = ['J','Ks']
    mosaicNames = ['scalar_J','scalar_Ks']
    titleText = [r"Scalar-fit $J$", r"Scalar-fit $K_s$"]
    for band, mosaicName, ax, title in zip(bands, mosaicNames, axes, titleText):
        resFlux = ScalarResidualFlux(mosaicName)
        relResiduals = resFlux.relative_residuals()
        fieldSelector = {"FILTER":band}
        couplingsGraph = CouplingsGraph(ax, fieldSelector)
        couplingsGraph.set_coupling_values(relResiduals)
        couplingsGraph.set_value_lim(0., 2.)
        label = r"$F_\mathrm{residual} / \sigma_\Delta$"
        couplingsGraph.render(cbar=True,
                clabel=label,
                title=title,
                cmult=0.5, cshrink=0.99)
    fig.savefig(plotPath+".pdf", format="pdf")
    fig.savefig(plotPath+".eps", format="eps")

def plot_scalar_residual_sb(plotPath):
    """docstring for compute_residual_scalar_flux"""
    fig = plt.figure(figsize=(6,4.2), frameon=False)
    fig.subplots_adjust(left=0.02, bottom=0.0, right=0.98, top=0.99,
            wspace=0.1, hspace=None)
    axes = [fig.add_subplot(121, frame_on=False),
            fig.add_subplot(122, frame_on=False)]
    bands = ['J','Ks']
    mosaicNames = ['scalar_J','scalar_Ks']
    titleText = [r"Scalar-fit $J$", r"Scalar-fit $K_s$"]
    for band, mosaicName, ax, title in zip(bands, mosaicNames, axes, titleText):
        resFlux = ScalarResidualFlux(mosaicName)
        resLevels = resFlux.residual_levels_sb(pixScale=0.3)
        fieldSelector = {"FILTER":band}
        couplingsGraph = CouplingsGraph(ax, fieldSelector)
        couplingsGraph.set_coupling_values(resLevels)
        if band == "J":
            label = r"$\mu_{\mathrm{residual},J}$ (mag arcsec$^{-2}$)"
        else:
            label = r"$\mu_{\mathrm{residual},{K_s}}$ (mag arcsec$^{-2}$)"
        couplingsGraph.render(cbar=True,
                clabel=label,
                title=title,
                cmult=1, cshrink=0.99)
    fig.savefig(plotPath+".pdf", format="pdf")
    fig.savefig(plotPath+".eps", format="eps")

class CouplingsGraph(object):
    """Base class for plotting inter-block couplings with some intensity."""
    def __init__(self, ax, fieldSelector, scaleFactor=2.):
        super(CouplingsGraph, self).__init__()
        self.ax = ax
        self.footprintDB = FootprintDB()
        self.selector = fieldSelector # footprintDB query dictionary
        self.values = None
        self.couplingNames = None
        self.fieldnames = None
        self.scaleFactor = scaleFactor
        self.minThick = 1.
        self.maxThick = 8.
    
    def set_coupling_values(self, values):
        """`values` is a dict of scalars, the keys are tuples of fieldnames
        in the footprintDB.
        """
        self.values = values
        self.couplingNames = self.values.keys()

        a = np.array([v for f,v in self.values.iteritems()])
        self.vmin = a.min()
        self.vmax = a.max()

        self.fieldnames = []
        for field1,field2 in self.couplingNames:
            self.fieldnames.append(field1)
            self.fieldnames.append(field2)

    def set_value_lim(self, zmin, zmax):
        """Assign limits on the colour bar."""
        self.vmin = zmin
        self.vmax = zmax

    def render(self, cbar=None, cax=None, clabel="", title="",
            cmult=None, cshrink=1.):
        """Renders the coupling network into the axes."""
        self._init_colour_bar()

        # Define the field polygons
        self.allFields = []
        origCenters = []
        origPolygons = []
        for fieldname in self.fieldnames:
            sel = self.selector
            sel['field'] = fieldname.split("_")[0]
            doc = self.footprintDB.find(sel)[0]
            fieldname, verts, center = self._parse_field_doc(doc)
            self.allFields.append(fieldname)
            origCenters.append(center)
            origPolygons.append(verts)
        explodedCenters = self._explode_field_centers(origCenters)
        explodedBoxes = self._explode_field_boxes(explodedCenters, origCenters,
                origPolygons)
        xlim, ylim = self._get_range(explodedBoxes)

        for verts in explodedBoxes:
            polyPatch = mpl.patches.Polygon(verts, color=(0.8,0.8,0.8,0.5),
                    closed=True, fill=True, ec='k', lw=0.5, zorder=1)
            self.ax.add_patch(polyPatch)
        

        # Define the couplings
        boxCenters = dict(zip(self.fieldnames, explodedCenters))
        for pairKey, value in self.values.iteritems():
            srcNode = str(pairKey[0])
            dstNode = str(pairKey[1])
            srcCoord = boxCenters[srcNode]
            dstCoord = boxCenters[dstNode]
            edgeColour = self._link_colour(value)
            edgeThickness = self._link_thickness(value)
            edgePatch = mpl.patches.ConnectionPatch(srcCoord, dstCoord,
                'data', coordsB=None,
                axesA=None, axesB=None, arrowstyle='-', arrow_transmuter=None,
                connectionstyle='arc3', connector=None, patchA=None, patchB=None,
                shrinkA=0.0, shrinkB=0.0, mutation_scale=10.0,
                mutation_aspect=None, clip_on=False,
                lw=edgeThickness, ec=edgeColour, zorder=10)
            self.ax.add_patch(edgePatch)

        if cbar:
            cb = plt.colorbar(self.mappable, cax=cax, ax=self.ax, pad=0,
                    orientation='horizontal', shrink=cshrink)
            cb.set_label(clabel)
            if cmult is not None:
                cb.locator = mpl.ticker.MultipleLocator(base=cmult)
                cb.update_ticks()

        self.ax.set_xlim(max(xlim),min(xlim))
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal')
        self.ax.set_frame_on('False')

        for label in self.ax.xaxis.get_majorticklabels():
            label.set_visible(False)
        for label in self.ax.yaxis.get_majorticklabels():
            label.set_visible(False)
        for tick in self.ax.xaxis.get_major_ticks():
            tick.set_visible(False)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.set_visible(False)

        if title is not "":
            self.ax.text(0.9,0.9,title,ha="right",va="top",
                    transform=self.ax.transAxes)

    def _parse_field_doc(self, fieldDoc):
        """Extracts information from a field doc. Returns a list of

        1. field name
        2. list of (xi,eta) vertices
        3. xi0, eta0 center of footprint
        """
        fieldname = fieldDoc['field']
        radec = fieldDoc['radec_poly']
        verts = []
        allXi = []
        allEta = []
        for (ra,dec) in radec:
            v = equatorialToTan(ra,dec, M31RA0,M31DEC0)
            allXi.append(v[0])
            allEta.append(v[1])
            verts.append(v)
        allXi = np.array(allXi)
        allEta = np.array(allEta)
        xi0 = allXi.mean()
        eta0 = allEta.mean()
        return fieldname, verts, (xi0,eta0)

    def _explode_field_centers(self, fieldCenters):
        """Given a list of field centers in xi0, eta0, compute centers
        of exploded fields"""
        newCenters = []
        for xy in fieldCenters:
            x, y = xy
            xy = np.array(xy)
            d = np.sqrt(x**2. + y**2.)
            v = xy / d
            xyPrime = v* (d + self.scaleFactor*d)
            newCenters.append((xyPrime[0], xyPrime[1]))
        return newCenters

    def _explode_field_boxes(self, fieldCenters, origCenters, origPolygons):
        """Given a list field centers, compute polygon vertices of exploded
        fields.
        """
        newPolygons = []
        for xy, origXY, origPolygon in zip(fieldCenters, origCenters, origPolygons):
            deltaX = xy[0] - origXY[0]
            deltaY = xy[1] - origXY[1]
            newPolygon = [(vert[0]+deltaX, vert[1]+deltaY) for vert in origPolygon]
            newPolygons.append(newPolygon)
        return newPolygons

    def _get_range(self, fieldBoxes):
        """Returns tuples of xlim, ylim that cover the domain of the field boxes."""
        xverts = []
        yverts = []
        for box in fieldBoxes:
            for vert in box:
                xverts.append(vert[0])
                yverts.append(vert[1])
        xlim = (min(xverts), max(xverts))
        ylim = (min(yverts), max(yverts))
        return xlim, ylim

    # self._init_colour_bar()
    def _init_colour_bar(self):
        """Creates teh mappable colourbar object."""
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax,
                clip=False)
        self.mappable = mpl.cm.ScalarMappable(norm=norm,
                cmap=mpl.cm.jet)
        self.mappable.set_array([v for f,v in self.values.iteritems()])
        self.mappable.set_clim(vmin=self.vmin, vmax=self.vmax)
        print "clim:", self.vmin, self.vmax

    def _link_thickness(self, v):
        """docstring for _link_thickness"""
        if v > self.vmax: v = self.vmax
        elif v < self.vmin: v = self.vmax

        normVal = (v-self.vmin)/self.vmax # value between 0. and 1.
        thickness = normVal * (self.maxThick - self.minThick) + self.minThick
        return thickness

    def _link_colour(self, v):
        """docstring for _link_thickness"""
        return self.mappable.to_rgba(v)



if __name__ == '__main__':
    main()


