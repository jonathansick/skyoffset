#!/usr/bin/env python
# encoding: utf-8
"""
For creating blocks from individual frames of of several aray camera exposures.

Here we assume a sky bias only between integrations,  but the all frames
within an integration see the same sky. Thus each exposure has a single sky
offset applied to all of its frames while making a block. This effectively
bipasses stack production. Our procedure is:

- In each 'stack', compute the usual frame sky offsets.
- For each image, compute the median of those frame sky offsets.
- Apply median sky offsets to each frame.
- Produce *entire* block with Swarp.

Use this module in cases where you have confidence in sky background uniformity
across single images.

.. note:: Currently we just run ChipStacker to solve offsets.
"""

import os
import shutil
import multiprocessing

import numpy as np
import astropy.io.fits

import moastro.astromatic

from skyoffset.stackfactory import ChipStacker
import skyoffset.offsettools as offset_tools
from skyoffset.noisefactory import NoiseMapFactory


class LockedBlockFactory(object):
    """Pipeline for creating blocks if we assume all frames in a given
    integration share a common zeropoint.

    Parameters
    ----------
    block_name : str
        The `_id` for this block.
    imagelog : :class:`moastro.imagelog.ImageLog` instance.
        Image log with references to the resampled frames.
    stackdb : :class:`skyoffset.imagedb.MosaicDB` instance
        The MosaicDB instance to store stack documents in (used temporarily).
    blockdb : :class:`skyoffset.imagedb.MosaicDB` instance
        The MosaicDB instance to store block documents in.
    image_sel : dict
        ImageLog selector for images to produce a block from.
    workdir : str
        Directory to make blocks in. This directory will be created if
        necessary.
    swarp_configs : dict
        A dictionary of configurations to pass to
        :class:`moastro.astromatic.Swarp`.
    db_meta : dict
        Optional dicionary of metadate to save with this block's document
    """
    def __init__(self, block_name, imagelog, stackdb, blockdb, image_sel,
            workdir, swarp_configs=None, db_meta=None):
        super(LockedBlockFactory, self).__init__()
        self.block_name = block_name
        self.workdir = workdir
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        self.stackdb = stackdb
        self.imagelog = imagelog
        self.blockdb = blockdb
        self.image_sel = dict(image_sel)
        if swarp_configs:
            self._swarp_configs = dict(swarp_configs)
        else:
            self._swarp_configs = {}
        self.db_meta = db_meta
        self.noise_path = None

    def make_stack(self, stack_name, image_key, weight_key, db_meta=None,
            clean_files=True, n_iter_max=4):
        """Make a stack simply to get frame sky offsets."""
        stackdir = os.path.join(self.workdir, stack_name)
        image_keys = []
        image_paths = []
        weight_paths = []
        s = dict(self.image_sel)
        s[image_key] = {"$exists": 1}
        s[weight_key] = {"$exists": 1}
        docs = self.imagelog.find(s, fields=[image_key, weight_key])
        assert docs.count() > 0
        for doc in docs:
            image_keys.append(doc['_id'])
            image_paths.append(doc[image_key])
            weight_paths.append(doc[weight_key])
        stacker = ChipStacker(self.stackdb, stackdir,
            swarp_configs=self._swarp_configs)
        stacker.pipeline(stack_name, image_keys, image_paths, weight_paths,
                db_meta=db_meta, n_iter_max=n_iter_max)
        if clean_files:
            shutil.rmtree(stackdir)

    def estimate_offsets(self, stack_sel):
        """Estimate single sky offsets for each camera exposure as the
        median frame offset estimated in individual stacks.
        """
        # Hold arrays of frame offsets observed for each image
        offset_ests = {}
        for stack in self.stackdb.find(stack_sel):
            for ik, offset in stack['offsets'].iteritems():
                if ik in offset_ests:
                    offset_ests[ik].append(offset)
                else:
                    offset_ests[ik] = [offset]
        self.offsets = {}
        self.offset_std = {}
        for ik, offsets in offset_ests.iteritems():
            ests = np.array(offsets)
            self.offsets[ik] = np.median(ests)
            self.offset_std[ik] = np.std(ests)
        # Normalize offsets
        net_offset = np.mean([d for ik, d in self.offsets.iteritems()])
        for ik, offset in self.offsets.iteritems():
            self.offsets[ik] = offset - net_offset

    def make_mosaic(self, image_path_keys, weight_path_keys,
            noise_path_keys=None,
            threads=multiprocessing.cpu_count(),
            delete_offset_images=True,
            target_fits=None):
        """Create the mosaic image using offsets computed with
        :class:`estimate_offsets`.
        
        Parameters
        ----------
        image_path_keys : list
            Sequence of keys into ImageLog for resampled images. This is a
            sequence since multi-extension FITS get split by Swarp when
            resampling.
        weight_path_keys : list
            Counterpart to `image_path_keys` for resampled weight maps.
            Must have the same order as `image_path_keys`.
        target_fits : str
            Set to the path of a FITS file that will be used to define the
            output frame of the block. The output blocks will then correspond
            pixel-to-pixel. Note that both blocks should already be resampled
            into the same pixel space.
        """
        image_paths = []
        weight_paths = []
        offset_paths = []
        offsets = []
        args = []
        s = dict(self.image_sel)
        for ikey, wkey in zip(image_path_keys, weight_path_keys):
            s[ikey] = {"$exists": 1}
            s[wkey] = {"$exists": 1}
        docs = self.imagelog.find(s)
        assert docs.count() > 0
        for doc in docs:
            for ikey, wkey in zip(image_path_keys, weight_path_keys):
                image_paths.append(doc[ikey])
                weight_paths.append(doc[wkey])
                offset_paths.append(os.path.join(self.workdir,
                        os.path.basename(doc[ikey])))
                offsets.append(self.offsets[doc['_id']])
                arg = (doc['_id'], image_paths[-1],
                    offsets[-1], offset_paths[-1])
                args.append(arg)
        if threads > 1:
            map(offset_tools.apply_offset, args)
        else:
            pool = multiprocessing.Pool(processes=threads)
            pool.map(offset_tools.apply_offset, args)

        swarp_configs = dict(self._swarp_configs)
        swarp_configs.update({"RESAMPLE": "N", "COMBINE": "Y"})
        swarp = moastro.astromatic.Swarp(offset_paths, self.block_name,
                weightPaths=weight_paths, configs=swarp_configs,
                workDir=self.workdir)
        if target_fits and os.path.exists(target_fits):
            swarp.set_target_fits(target_fits)
        swarp.run()
        block_path, weight_path = swarp.mosaic_paths()

        # Make noisemap if possible
        noise_paths = []
        if noise_path_keys is not None:
            s = dict(self.image_sel)
            for ikey, wkey, nkey in zip(image_path_keys, weight_path_keys,
                    noise_path_keys):
                s[ikey] = {"$exists": 1}
                s[wkey] = {"$exists": 1}
                s[nkey] = {"$exists": 1}
            docs = self.imagelog.find(s)
            for doc in docs:
                for nkey in noise_path_keys:
                    noise_paths.append(doc[nkey])
            self._make_noisemap(noise_paths, weight_paths, block_path)

        if delete_offset_images:
            for p in offset_paths:
                if os.path.exists(p):
                    os.remove(p)

        # Save document to BlockDB
        doc = {}
        if self.db_meta:
            doc.update(self.db_meta)
        doc['_id'] = self.block_name
        doc['image_path'] = block_path
        doc['weight_path'] = weight_path
        doc['offsets'] = self.offsets
        doc['offset_sigmas'] = self.offset_std
        if self.noise_path is not None:
            doc['noise_path'] = self.noise_path
        self.blockdb.c.save(doc)

        self.blockdb.add_footprint_from_header(self.block_name,
            astropy.io.fits.getheader(block_path))

    def _make_noisemap(self, noise_paths, weight_paths, mosaic_path):
        """Make a noise map for this coadd given noisemaps of individual
        images.
        """
        factory = NoiseMapFactory(noise_paths, weight_paths, mosaic_path,
                swarp_configs=dict(self._swarp_configs),
                delete_temps=False)
        self.noise_path = factory.map_path
