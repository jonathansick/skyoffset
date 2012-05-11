import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from owl.Footprint import equatorialToTan
from andpipe.footprintdb import FootprintDB
from mosaicdb import MosaicDB
from stackdb import StackDB
from andpipe.imagelog import WIRLog

M31RA0 = 10.6846833
M31DEC0 = 41.2690361

def map_offsetratio():
    plotPath = "skysubpub/offset_ratio_map"

    fig = plt.figure(figsize=(6,3))
    axJ = fig.add_subplot(121)
    axK = fig.add_subplot(122)
    axes = {"J": axJ, "Ks": axK}
    
    for band, ax in axes.iteritems():
        offsetRatios = compute_offset_ratios(band)
        
        s = {"instrument": "WIRCam", "kind": "ph2"}
        plotter = FieldLevelPlotter(ax, s)
        plotter.set_field_values(offsetRatios)
        if band == "J": cbar = False
        else: cbar = True
        plotter.render(vmin=0.1, vmax=1.5, cbar=cbar,
                clabel=r"$|\Delta_\mathrm{block} / \sigma_\mathrm{stack}|$")
        ax.set_xlim(1.1,-1.1)
        ax.set_ylim(-1.5,1.5)
        ax.set_aspect('equal')

        if band == "J":
            ax.text(0.1,0.1, r"$J$", transform=ax.transAxes)
        elif band == "Ks":
            ax.text(0.1,0.1, r"$K_s$", transform=ax.transAxes)

    for label in axes['Ks'].get_yaxis().get_majorticklabels():
        label.set_visible(False)
    
    fig.savefig(plotPath+".pdf", format="pdf")
    fig.savefig(plotPath+".eps", format="eps")

def map_offsetratio_enhanced():
    """Includes 09/07B histograms and the fieldmap."""
    #07: 31, 120, 180;
    #09: 178, 223, 138; 
    c07 = (.121568627, .470588235, .705882353, 1.)
    c09 = (.698039216, .874509804, .541176471, 1.)
    plotPath = "skysubpub/offset_ratio_map_enhanced"
    fig = plt.figure(figsize=(6,3))
    mapAxes = {"J": fig.add_axes([0.28,0.15,.325,0.76], frameon=False),
            "Ks": fig.add_axes([0.23+.325,0.15,.325,0.76], frameon=False)}
    histAxes = {"J": fig.add_axes([0.06,0.15,.2,0.38]),
            "Ks": fig.add_axes([0.06,0.57,.2,0.38])}
    axCbar = fig.add_axes([0.88,0.15,0.02,0.76])

    for band, ax in mapAxes.iteritems():
        offsetRatios = compute_offset_ratios(band,
                selector={"OBJECT":{"$nin":["M31-44","M31-45","M31-46"]}})
        
        s = {"instrument": "WIRCam", "kind": "ph2"}
        plotter = FieldLevelPlotter(ax, s)
        plotter.set_field_values(offsetRatios)
        if band == "J": cbar = False
        else: cbar = True
        plotter.render(vmin=0.1, vmax=1.5, cbar=cbar,
                clabel=r"$|\Delta_\mathrm{block} / \sigma_\mathrm{stack}|$",
                cax=axCbar)
        ax.set_xlim(1.1,-1.1)
        ax.set_ylim(-1.25,1.25)
        ax.set_aspect('equal')

        for label in ax.get_yaxis().get_majorticklabels(): label.set_visible(False)
        for label in ax.get_xaxis().get_majorticklabels(): label.set_visible(False)
        for tick in ax.get_yaxis().get_major_ticks(): tick.set_visible(False)
        for tick in ax.get_xaxis().get_major_ticks(): tick.set_visible(False)

        if band == "J":
            ax.text(0.6,0.9, r"$J$", transform=ax.transAxes)
        elif band == "Ks":
            ax.text(0.6,0.9, r"$K_s$", transform=ax.transAxes)

    #for label in mapAxes['Ks'].get_yaxis().get_majorticklabels():
    #    label.set_visible(False)

    for band, ax in histAxes.iteritems():
        asarr = lambda d: np.array([v for k,v in d.iteritems()])
        ratios07 = asarr(compute_offset_ratios(band,
            selector={"RUNID": {"$in": ["07BC20","07BH47"]},
                "OBJECT":{"$nin":["M31-44","M31-45","M31-46"]}}))
        ratios09 = asarr(compute_offset_ratios(band,
            selector={"RUNID": {"$in": ["09BC29","09BH52"]},
                "OBJECT":{"$nin":["M31-44","M31-45","M31-46"]}}))
        bins = np.arange(0.,3.,0.2)
        ax.hist(ratios07, histtype='step', bins=bins, ec=c07)
        ax.hist(ratios09, histtype='step', bins=bins, ec=c09)
        if band == "J":
            ax.text(0.8,0.8, r"$J$", transform=ax.transAxes)
        elif band == "Ks":
            ax.text(0.8,0.8, r"$K_s$", transform=ax.transAxes)
    
    histAxes['J'].set_xlim(0.,1.5)
    histAxes['Ks'].set_xlim(0.,1.5)
    histAxes['J'].set_xticks([0.0,0.5,1.,1.5])
    histAxes['Ks'].set_xticks([0.0,0.5,1.,1.5])
    histAxes['J'].yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2))
    histAxes['Ks'].yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2))
    histAxes['J'].set_ylabel("number of blocks")
    histAxes['J'].set_xlabel(r"$|\Delta_\mathrm{block} / \sigma_\mathrm{stack}|$")
    for label in histAxes['Ks'].get_xaxis().get_majorticklabels():
        label.set_visible(False)
    #histAxes['Ks'].set_ylabel("number of blocks")
    
    fig.savefig(plotPath+".pdf", format="pdf")
    fig.savefig(plotPath+".eps", format="eps")


def compute_offset_ratios(band, selector={}):
    """Compute the ratio of total offset for a block against the
    dispersion of offsets in individual stacks.
    """
    stackDB = StackDB(dbname="m31", cname="wircam_stacks")
    imageLog = WIRLog()
    selector.update({"TYPE":"sci"})
    allFields = imageLog.find_unique("OBJECT",
            selector=selector)
    #allFields.remove("M31-44")
    #allFields.remove("M31-45")
    #allFields.remove("M31-46")

    frameSigmas = {}
    for field in allFields:
        s = {"OBJECT":field, "FILTER":band}
        docs = stackDB.find_stacks(s)
        offsetSigma = []
        for stackName, doc in docs.iteritems():
            offsets = np.array([doc['offsets'][k] for k in doc['offsets']])
            offsetSigma.append(offsets.std())
        frameSigmas[field] = min(offsetSigma)
    print "Frame sigmas", frameSigmas
    # Now find the largest offset from a block to the mosaic
    mosaicDB = MosaicDB()
    mosaicName = "scalar_%s" % band
    doc = mosaicDB.get_mosaic_doc(mosaicName)
    blockOffsets = doc['offsets']

    offsetRatios = {}
    print "all fields:", allFields
    #for field, blockOffset in blockOffsets.iteritems():
    for objname in allFields:
        
        #objname = field.split("_")[0]
        blockOffset = blockOffsets[objname+"_"+band]
        #if field not in allFields: continue
        print ":", objname, blockOffset

        offsetRatio = math.fabs(blockOffset / frameSigmas[objname])
        offsetRatios[objname] = offsetRatio
    return offsetRatios


class FieldLevelPlotter(object):
    """Plots a fieldmap in the given axes, prescribing the values in field
    boxes as the mapped colour
    """
    def __init__(self, ax, selector={}):
        super(FieldLevelPlotter, self).__init__()
        self.ax = ax
        self.footprintDB = FootprintDB()
        self.selector = selector
        self.values = None
        self.fields = None
    
    def set_field_values(self, values):
        """`values` is a dict of scalars, the keys are fieldnames"""
        self.values = values
        self.fields = self.values.keys()

        a = np.array([v for f,v in self.values.iteritems()])
        self.zmin = a.min()
        self.zmax = a.max()

    def set_value_lim(self, zmin, zmax):
        """Assign limits on the colour bar"""
        self.zmin = zmin
        self.zmax = zmax

    def render(self, vmin=None, vmax=None, cbar=False, clabel="",
            cax=None, cbarOrientation="vertical", cticks=None,
            borderColour='k', borderWidth=0.5, zorder=1):
        """Draw the fields with values."""
        allVerts = []
        allValues = []
        
        fields = self.values.keys()
        fields.sort()
        fieldNums = [int(field.split("-")[1]) for field in fields]
        fieldNums.sort()
        fields = ["M31-%i"%n for n in fieldNums]
        #for field, value in self.values.iteritems():
        for field in fields:
            value = self.values[field]
            sel = self.selector
            sel['field'] = field
            print sel
            doc = self.footprintDB.find(sel)[0]
            print doc
            fieldname, verts, center = self._parse_field_doc(doc)
            allVerts.append(verts)
            allValues.append(value)
        
        if vmin == None or vmax == None:
            vmin = min(allValues)
            vmax = max(allValues)
        
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        mappable = mpl.cm.ScalarMappable(norm=norm,
                cmap=mpl.cm.jet)
        mappable.set_array(allValues)

        for verts, value in zip(allVerts, allValues):
            c = mappable.to_rgba(value)
            polyPatch = mpl.patches.Polygon(verts, color=c, closed=True,
            fill=True, ec=borderColour, lw=borderWidth, zorder=zorder)
            self.ax.add_patch(polyPatch)
        
        if cbar:
            cb = plt.colorbar(mappable, cax=cax, ax=self.ax,
                    orientation=cbarOrientation,
                    ) # cbarOrientation
            cb.set_label(clabel)
            if cticks is not None:
                cb.set_ticks(cticks)
            
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

if __name__ == '__main__':
    #map_offsetratio()
    map_offsetratio_enhanced()
