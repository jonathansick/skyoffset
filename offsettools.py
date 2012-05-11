import numpy
import pyfits

def apply_offset(arg):
    """Worker: Subtracts the DC offset from a frame, and save the new image
    to the given output path.
    """
    frame, imagePath, offset, outputPath = arg
    fits = pyfits.open(imagePath)
    image = fits[0].data
    goodPix = numpy.where(image != 0.)
    # image[image != 0.] = image[image != 0.] - offset
    print frame,
    print offset
    image[goodPix] = image[goodPix] - offset # CHANGED
    fits.writeto(outputPath, clobber=True)
    return frame, outputPath

def apply_planar_offset(arg):
    """Subtracts the planar offset from a frame, and saves the new image
    to the given output path.
    """
    frame, imagePath, offset, outputPath = arg

    fits = pyfits.open(imagePath)
    image = fits[0].data
    zeroPix = numpy.where(image == 0.)
    
    # Make a mock image from the plane; and subtract it.
    shape = image.shape
    offsetImage = make_plane_image(shape, offset)
    
    newImage = image - offsetImage
    newImage[zeroPix] = 0.
    fits[0].data = newImage
    
    # print offsetPath
    fits.writeto(outputPath, clobber=True)

def make_plane_image(shape, plane):
    """Given the tuple shape (ysize,xsize), """
    ysize, xsize = shape
    y0 = int(ysize/2.)
    x0 = int(xsize/2.)
    
    xCoords = []
    yCoords = []
    xIndices = []
    yIndices = []
    for i in xrange(xsize):
        x = i - x0
        for j in xrange(ysize):
            #y = -(j - y0)
            y = j - y0 # CHANGED
            xCoords.append(x)
            yCoords.append(y)
            xIndices.append(i)
            yIndices.append(j)
    xCoords = numpy.array(xCoords)
    yCoords = numpy.array(yCoords)
    xIndices = numpy.array(xIndices, dtype=int)
    yIndices = numpy.array(yIndices, dtype=int)
    
    mx, my, c = plane
    offsetImage = numpy.zeros(shape)
    offsetImage[yIndices, xIndices] = mx*xCoords + my*yCoords + c
    return offsetImage

