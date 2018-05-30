# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from ..capsule import Module
from ..utils.caps_utils import mixed_shape, equal_shape
from .. import constraints, regularizers

from keras import initializers
from keras import backend as K
from keras.engine.topology import InputSpec

import numpy as np


class LinearTransformation(Module):
    def __init__(self,
                 output_dim=None,
                 axis=-1,
                 scope='local',
                 linear_map_initializer='glorot_uniform',
                 linear_map_regularizer=None,
                 linear_map_constraint=None,
                 **kwargs):
        """
        Not a real module!

        :param output_dim: None input_shape == output_shape
        :param linear_map_initializer:
        :param linear_map_regularizer:
        :param linear_map_constraint:
        :param scope:
        :param kwargs:
        """
        if isinstance(output_dim, int) or output_dim is None:
            self._output_dim = output_dim
        else:
            raise ValueError("output_dim must be int or None. You provide: " +
                             str(output_dim))
        self.output_dim = None

        # scope: local, global, capsule_wise (defines the apply scope of the transformation)
        if scope in ('local', 'global', 'capsule_wise', 'channel_wise'):
            self.scope = scope
        else:
            raise ValueError("scope must be 'local', 'global', 'capsule_wise' or 'channel_wise': You provide: "
                             + str(scope))

        if isinstance(axis, int):
            if axis in (0, 1, 2):
                raise ValueError("It's not allowed to apply transformations over batch (0), capsule (1) or channel (2) "
                                 "dimension. You provide: " + str(axis))
            else:
                self.axis = axis
        else:
            raise TypeError("axis must be integer. You provide: " + str(axis))

        self.linear_map_initializer = initializers.get(linear_map_initializer)
        self.linear_map_regularizer = regularizers.get(linear_map_regularizer)
        self.linear_map_constraint = constraints.get(linear_map_constraint)
        self.linear_maps = None

        self._num_maps = None

        # be sure to call this somewhere
        super(LinearTransformation, self).__init__(module_input=True,
                                                   module_output=True,
                                                   need_capsule_params=False,
                                                   **kwargs)

    def _build(self, input_shape):
        if self.axis <= -1:
            self.axis = len(input_shape[0]) + self.axis

        if self.axis in (0, 1, 2):
            raise ValueError("It's not allowed to apply transformations over batch (0), capsule (1) or channel (2) "
                             "dimension. You provide: " + str(self.axis))

        if self.axis > len(input_shape[0])-1:
            raise ValueError("assigned axis is out of signal shape: axis=" + str(self.axis) + " is not in "
                             "signal_shape=" + str(input_shape[0]))

        if not self.built:
            # compute shape of linear_maps
            if self.scope == 'local':
                self._num_maps = tuple(input_shape[0][1:3])
            elif self.scope == 'global':
                self._num_maps = ()
            elif self.scope == 'channel_wise':
                self._num_maps = (input_shape[0][2],)
            else:  # capsule_wise
                self._num_maps = (input_shape[0][1],)

            self.output_dim = self._compute_output_shape(input_shape)[0][self.axis]
            linear_map_shape = self._num_maps + (input_shape[0][self.axis], self.output_dim)

            # transposed matrix (we compute x^T * A^T instead of A * x, because the signal is always given in the form
            # x^T)
            self.linear_maps = self.add_weight(shape=linear_map_shape,
                                               initializer=self.linear_map_initializer,
                                               regularizer=self.linear_map_regularizer,
                                               constraint=self.linear_map_constraint,
                                               name='linear_maps')

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        # inverse permutation
        def inv_perm(perm):
            inverse = [0] * len(perm)
            for i, p in enumerate(perm):
                inverse[p] = i
            return inverse

        # signal is dict: extract signal from diss
        signals = inputs[0]

        signal_shape = mixed_shape(signals)
        ndim = self.input_spec[0].ndim
        atom_axes = list(range(3, ndim))
        atom_axes.remove(self.axis)

        if self.scope == 'local':
            with K.name_scope('signal_preprocessing'):
                perm = [1, 2, self.axis, 0] + atom_axes
                signals = K.permute_dimensions(signals, perm)

                if ndim > 4:
                    signals = K.reshape(signals, [signal_shape[1], signal_shape[2], signal_shape[self.axis], -1])

            with K.name_scope('linear_mapping'):
                # multiply over all batches by using the Theano behavior
                signals = K.batch_dot(self.linear_maps, signals, axes=[2, 2])

            with K.name_scope('signal_postprocessing'):
                if ndim > 4:
                    signals = K.reshape(signals, [signal_shape[1], signal_shape[2], self.output_dim,
                                                  signal_shape[0]] + [signal_shape[i] for i in atom_axes])

                signals = K.permute_dimensions(signals, inv_perm(perm))

        elif self.scope == 'global':
            with K.name_scope('signal_preprocessing'):
                dims = list(range(ndim))
                dims.remove(self.axis)
                perm = dims + [self.axis]

                signals = K.permute_dimensions(signals, perm)

            with K.name_scope('linear_mapping'):
                signals = K.dot(signals, self.linear_maps)

            with K.name_scope('signal_postprocessing'):
                signals = K.permute_dimensions(signals, inv_perm(perm))

        elif self.scope == 'channel_wise':
            with K.name_scope('signal_preprocessing'):
                perm = [2, self.axis, 0, 1] + atom_axes
                signals = K.permute_dimensions(signals, perm)

                signals = K.reshape(signals, [signal_shape[2], signal_shape[self.axis], -1])

            with K.name_scope('linear_mapping'):
                # multiply over all batches by using the Theano behavior
                signals = K.batch_dot(self.linear_maps, signals, axes=[1, 1])

            with K.name_scope('signal_postprocessing'):
                signals = K.reshape(signals, [signal_shape[2], self.output_dim, signal_shape[0], signal_shape[1]]
                                    + [signal_shape[i] for i in atom_axes])

                signals = K.permute_dimensions(signals, inv_perm(perm))

        else:  # capsule_wise
            with K.name_scope('signal_preprocessing'):
                perm = [1, self.axis, 0, 2] + atom_axes
                signals = K.permute_dimensions(signals, perm)

                signals = K.reshape(signals, [signal_shape[1], signal_shape[self.axis], -1])

            with K.name_scope('linear_mapping'):
                # multiply over all batches by using the Theano behavior
                signals = K.batch_dot(self.linear_maps, signals, axes=[1, 1])

            with K.name_scope('signal_postprocessing'):
                signals = K.reshape(signals, [signal_shape[1], self.output_dim, signal_shape[0], signal_shape[2]]
                                    + [signal_shape[i] for i in atom_axes])

                signals = K.permute_dimensions(signals, inv_perm(perm))

        inputs[0] = signals
        return inputs

    def _compute_output_shape(self, input_shape):
        if not self.built:
            if self._output_dim is None:
                output_dim = input_shape[0][self.axis]
            else:
                output_dim = self._output_dim

            signal_shape = list(input_shape[0])
            signal_shape[self.axis] = output_dim

        else:
            signal_shape = list(input_shape[0])
            signal_shape[self.axis] = self.output_dim

        return [tuple(signal_shape), tuple(input_shape[1])]

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'axis': self.axis,
                  'scope': self.scope,
                  'linear_map_initializer': initializers.serialize(self.linear_map_initializer),
                  'linear_map_regularizer': regularizers.serialize(self.linear_map_regularizer),
                  'linear_map_constraint': constraints.serialize(self.linear_map_constraint)}
        super_config = super(LinearTransformation, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class AddBias(Module):
    def __init__(self,
                 axes=-1,
                 scope='local',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Not a real module!

        :param output_dim: None input_shape == output_shape
        :param linear_map_initializer:
        :param linear_map_regularizer:
        :param linear_map_constraint:
        :param scope:
        :param kwargs:
        """
        # scope: local, global, capsule_wise (defines the apply scope of the transformation)
        if scope in ('local', 'global', 'capsule_wise', 'channel_wise'):
            self.scope = scope
        else:
            raise ValueError("scope must be 'local', 'global', 'capsule_wise' or 'channel_wise': You provide: "
                             + str(scope))

        if isinstance(axes, (tuple, list)):
            axes = list(axes)
        elif isinstance(axes, int):
            axes = [axes]
        else:
            raise TypeError("axes must be list, tuple or int. You provide: " + str(type(axes)))

        for axis in axes:
            if axis in (0, 1, 2):
                raise ValueError("It's not allowed to apply transformations over batch (0), capsule (1) or channel (2) "
                                 "dimension. You provide: " + str(axes))
        self.axes = axes

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.biases = None

        self._num_bias = None

        # be sure to call this somewhere
        super(AddBias, self).__init__(module_input=True,
                                      module_output=True,
                                      need_capsule_params=False,
                                      **kwargs)

    def _build(self, input_shape):
        for i, axis in enumerate(self.axes):
            if axis <= -1:
                self.axes[i] = len(input_shape[0]) + axis

            if self.axes[i]in (0, 1, 2):
                raise ValueError("It's not allowed to apply transformations over batch (0), capsule (1) or channel (2) "
                                 "dimension. You provide: " + str(self.axes))

        self.axes = list(np.unique(np.array(self.axes, dtype=int)))

        if max(self.axes) > len(input_shape[0])-1:
            raise ValueError("An assigned axis is out of signal shape: axes=" + str(self.axes) + " is not in "
                             "signal_shape=" + str(input_shape[0]))

        if not self.built:
            # compute shape of linear_maps
            if self.scope == 'local':
                self._num_bias = tuple(input_shape[0][1:3])
            elif self.scope == 'global':
                self._num_bias = ()
            elif self.scope == 'channel_wise':
                self._num_bias = (input_shape[0][2],)
            else:  # capsule_wise
                self._num_bias = (input_shape[0][1],)

            bias_shape = self._num_bias + tuple([input_shape[0][axis] for axis in self.axes])

            # transposed matrix (we compute x^T * A^T instead of A * x, because the signal is always given in the form
            # x^T)
            self.biases = self.add_weight(shape=bias_shape,
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          name='biases')

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        # inverse permutation
        def inv_perm(perm):
            inverse = [0] * len(perm)
            for i, p in enumerate(perm):
                inverse[p] = i
            return inverse

        # signal is dict: extract signal from diss
        signals = inputs[0]

        ndim = self.input_spec[0].ndim
        atom_axes = list(range(3, ndim))
        batch_axes = []
        for axis in atom_axes:
            if axis not in self.axes:
                batch_axes.append(axis)

        if self.scope == 'local':
            perm = [0] + batch_axes + [1, 2] + self.axes
        elif self.scope == 'global':
            perm = [0, 1, 2] + batch_axes + self.axes
        elif self.scope == 'channel_wise':
            perm = [0, 1] + batch_axes + [2] + self.axes
        else:  # capsule_wise
            perm = [0, 2] + batch_axes + [1] + self.axes

        with K.name_scope('signal_preprocessing'):
            signals = K.permute_dimensions(signals, perm)

        with K.name_scope('linear_mapping'):
            signals = signals + self.biases

        with K.name_scope('signal_postprocessing'):
            signals = K.permute_dimensions(signals, inv_perm(perm))

        inputs[0] = signals
        return inputs

    def _compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'axes': self.axes,
                  'scope': self.scope,
                  'linear_map_initializer': initializers.serialize(self.bias_initializer),
                  'linear_map_regularizer': regularizers.serialize(self.bias_regularizer),
                  'linear_map_constraint': constraints.serialize(self.bias_constraint)}
        super_config = super(AddBias, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))
