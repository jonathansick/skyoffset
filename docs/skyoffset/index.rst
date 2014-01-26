The Skyoffset Documentation
===========================

Skyoffset is a python package for making wide-field mosaics of FITS images.
A principal feature of Skyoffset is the ability to produce a mosaic with a continuous background level by solving for sky offsets that minimize the intensity differences between overlapping images.

Skyoffset is developed by Jonathan Sick (Queen's University) for the Andromeda Optical and Infrared Disk Survey (ANDROIDS).
The Skyoffset algorithm was described in `Sick et al 2013 (arxiv:1303.6290) <http://arxiv.org/abs/1303.6290>`_.
A main feature of the Skyoffset algorithm is its handling of hierarchies, making it ideal for optimizing backgrounds in large mosaics made with array cameras (such as CFHT's MegaCam and WIRCam).

Skyoffset uses `MongoDB <http://www.mongodb.org>`_ in conjunction with `Mo'Astro <http://moastro.jonathansick.ca/moastro/>`_ to store metadata about each mosaic.
We use `Astromatic's Swarp <http://www.astromatic.net/software/swarp>`_ to handle image combination and propagate uncertainty maps.

Skyoffset is great for integrating into Python pipelines, with a convenient API and metadata storage in MongoDB.
If you're looking for an alternative package that requires less technological buy-in, `IPAC's Montage <http://montage.ipac.caltech.edu>`_ is great.
