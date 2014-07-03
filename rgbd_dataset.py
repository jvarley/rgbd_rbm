
from pylearn2.datasets import dense_design_matrix
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import numpy as np
import os
import cPickle
import IPython
from pylearn2.gui import patch_viewer
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided as ast



class RGBD(dense_design_matrix.DenseDesignMatrix):

    def __init__(self):
        def load_rgbd_data():
            data_path = "../data/pickled_rgbd/"

            out_total = np.zeros((0, 480,640,4))
            for file in os.listdir(data_path):
                print file
                data = cPickle.load(open(data_path + file))
                data_exp = np.expand_dims(data, 0)
                out_total = np.concatenate((out_total, data_exp),axis=0)

            num_images = out_total.shape[0]

            count = 0
            patches = np.zeros((num_images*480/8*640/8, 8, 8, 4))
            for image_index in range(out_total.shape[0]):
                for i in range(480/8):
                    for j in range(640/8):
                        patches[count] = out_total[image_index, i*8:(i+1)*8, j*8:(j+1)*8:]
                        count += 1

            patch_view = patch_viewer.PatchViewer((60,80), (8, 8))
            for patch_count in range(640/8*480/8):
                patch = patches[patch_count, :, :, 2]
                patch_view.add_patch(patch)
            #patch_view.show()

            count = 0
            flat_patches = np.zeros((out_total.shape[0]*480/8*640/8, 256))
            for patch_count in range(num_images*640/8*480/8):
                patch = patches[patch_count,:,:,:].flatten()
                flat_patches[count] = patch
                count += 1

            #IPython.embed()
            return flat_patches



        self.X = load_rgbd_data()
        self.y = None
        self.compress = False
        #self.view_converter = RGDBViewConverter((8,8,1))
        self.view_converter = RGDBViewConverter((16,16,1))

        super(RGBD, self).__init__(X=self.X, view_converter=self.view_converter)


class RGDBViewConverter(object):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    shape : WRITEME
    axes : WRITEME
    """
    def __init__(self, shape, axes=('b', 0, 1, 'c')):
        self.shape = shape
        self.pixels_per_channel = 1
        for dim in self.shape[:-1]:
            self.pixels_per_channel *= dim
        self.axes = axes
        self._update_topo_space()

    def view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.shape

    def weights_view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.shape

    def design_mat_to_topo_view(self, X):
        """
        .. todo::

            WRITEME
        """
        assert len(X.shape) == 2
        batch_size = X.shape[0]

        channel_shape = [batch_size, self.shape[0], self.shape[1], 1]
        dimshuffle_args = [('b', 0, 1, 'c').index(axis) for axis in self.axes]
        if self.shape[-1] * self.pixels_per_channel != X.shape[1]:
            raise ValueError('View converter with ' + str(self.shape[-1]) +
                             ' channels and ' + str(self.pixels_per_channel) +
                             ' pixels per channel asked to convert design'
                             ' matrix with ' + str(X.shape[1]) + ' columns.')

        def get_channel(channel_index):
            start = self.pixels_per_channel * channel_index
            stop = self.pixels_per_channel * (channel_index + 1)
            data = X[:, start:stop]
            return data.reshape(*channel_shape).transpose(*dimshuffle_args)

        channels = [get_channel(i) for i in xrange(self.shape[-1])]

        channel_idx = self.axes.index('c')
        rval = np.concatenate(channels, axis=channel_idx)
        assert len(rval.shape) == len(self.shape) + 1
        return rval

    def design_mat_to_weights_view(self, X):
        """
        .. todo::

            WRITEME
        """
        rval = self.design_mat_to_topo_view(X)

        # weights view is always for display
        rval = np.transpose(rval, tuple(self.axes.index(axis)
                                        for axis in ('b', 0, 1, 'c')))

        return rval

    def topo_view_to_design_mat(self, V):
        """
        .. todo::

            WRITEME
        """

        V = V.transpose(self.axes.index('b'),
                        self.axes.index(0),
                        self.axes.index(1),
                        self.axes.index('c'))

        num_channels = self.shape[-1]
        if np.any(np.asarray(self.shape) != np.asarray(V.shape[1:])):
            raise ValueError('View converter for views of shape batch size '
                             'followed by ' + str(self.shape) +
                             ' given tensor of shape ' + str(V.shape))
        batch_size = V.shape[0]

        rval = np.zeros((batch_size, self.pixels_per_channel * num_channels),
                        dtype=V.dtype)

        for i in xrange(num_channels):
            ppc = self.pixels_per_channel
            rval[:, i * ppc:(i + 1) * ppc] = V[..., i].reshape(batch_size, ppc)
        assert rval.dtype == V.dtype

        return rval

    def get_formatted_batch(self, batch, dspace):
        """
        .. todo::

            WRITEME properly

        Reformat batch from the internal storage format into dspace.
        """
        if isinstance(dspace, VectorSpace):
            # If a VectorSpace is requested, batch should already be in that
            # space. We call np_format_as anyway, in case the batch needs to be
            # cast to dspace.dtype. This also validates the batch shape, to
            # check that it's a valid batch in dspace.
            return dspace.np_format_as(batch, dspace)
        elif isinstance(dspace, Conv2DSpace):
            # design_mat_to_topo_view will return a batch formatted
            # in a Conv2DSpace, but not necessarily the right one.
            topo_batch = self.design_mat_to_topo_view(batch)
            if self.topo_space.axes != self.axes:
                warnings.warn("It looks like %s.axes has been changed "
                              "directly, please use the set_axes() method "
                              "instead." % self.__class__.__name__)
                self._update_topo_space()

            return self.topo_space.np_format_as(topo_batch, dspace)
        else:
            raise ValueError("%s does not know how to format a batch into "
                             "%s of type %s."
                             % (self.__class__.__name__, dspace, type(dspace)))

    def __setstate__(self, d):
        """
        .. todo::

            WRITEME
        """
        # Patch old pickle files that don't have the axes attribute.
        if 'axes' not in d:
            d['axes'] = ['b', 0, 1, 'c']
        self.__dict__.update(d)

        # Same for topo_space
        if 'topo_space' not in self.__dict__:
            self._update_topo_space()

    def _update_topo_space(self):
        """Update self.topo_space from self.shape and self.axes"""
        rows, cols, channels = self.shape
        self.topo_space = Conv2DSpace(shape=(rows, cols),
                                      num_channels=channels,
                                      axes=self.axes)

    def set_axes(self, axes):
        """
        .. todo::

            WRITEME
        """
        self.axes = axes
        self._update_topo_space()


