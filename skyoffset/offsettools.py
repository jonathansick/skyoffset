import numpy
import astropy.io.fits


def apply_offset(arg):
    """Worker: Subtracts the DC offset from a frame, and save the new image
    to the given output path.
    """
    frame, imagePath, offset, outputPath = arg
    fits = astropy.io.fits.open(imagePath)
    image = fits[0].data
    goodPix = numpy.where(image != 0.)
    # image[image != 0.] = image[image != 0.] - offset
    print frame,
    print offset
    image[goodPix] = image[goodPix] - offset  # CHANGED
    fits.writeto(outputPath, clobber=True)
    return frame, outputPath
