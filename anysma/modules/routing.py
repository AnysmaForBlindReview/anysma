# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import initializers
from keras import backend as K
from keras.engine.topology import InputSpec

from ..utils.caps_utils import mixed_shape
from ..capsule import Module
from .. import probability_transformations as prob_trans
from .. import constraints, regularizers


class GibbsRouting(Module):
    # Todo: Test of gibbs with scaling
    def __init__(self,
                 beta_initializer='ones',
                 beta_regularizer=None,
                 beta_constraint='NonNeg',
                 diss_regularizer=None,
                 signal_regularizer=None,
                 norm_axis='channels',
                 **kwargs):

        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.beta = None

        if norm_axis not in ('capsules', 'channels', 1, 2, -1):
            raise ValueError("norm_axis must be 'capsules' or 'channels'. You provide: " + str(norm_axis))
        if norm_axis == 'capsules':
            self.norm_axis = 1
        elif norm_axis == 'channels':
            self.norm_axis = 2
        else:
            self.norm_axis = norm_axis

        self.diss_regularizer = regularizers.get(diss_regularizer)
        self.signal_regularizer = regularizers.get(signal_regularizer)

        # be sure to call this somewhere
        super(GibbsRouting, self).__init__(module_input=True,
                                           module_output=True,
                                           need_capsule_params=False,
                                           **kwargs)

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("The number of capsules must be equal to the number of prototypes. Necessary "
                                 "assumption for Gibbs Routing. You provide " + str(input_shape[0][1]) + "!="
                                 + str(input_shape[1][1]) + ". Maybe you forgot the calling of a measuring module.")

            # add additional dimension to use broadcasting
            self.beta = self.add_weight(shape=(input_shape[0][1], 1),
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        name='beta')

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        signals = inputs[0]
        diss = inputs[1]

        # reshape of signal to vector necessary
        if self.input_spec[0].ndim > 4:
            signal_shape = mixed_shape(signals)
            signals = K.reshape(signals, signal_shape[0:3] + (-1,))

        with K.name_scope('routing_probabilities'):
            coefficients = prob_trans.neg_softmax(diss * self.beta, axis=self.norm_axis, max_stabilization=True)

        with K.name_scope('signal_routing'):
            signals = K.batch_dot(coefficients, signals, [2, 2])

        with K.name_scope('dissimilarity_routing'):
            diss = K.squeeze(K.batch_dot(coefficients, K.expand_dims(diss, -1), [2, 2]), -1)

        # reshape to new output_vector_shape necessary?
        if self.input_spec[0].ndim > 4:
            signals = K.reshape(signals, signal_shape[0:2] + signal_shape[3:])

        if self.diss_regularizer is not None:
            self.add_loss([self.diss_regularizer(diss)], inputs)
        if self.signal_regularizer is not None:
            self.add_loss([self.signal_regularizer(signals)], inputs)

        return {0: signals, 1: diss}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        del signals[2]
        del diss[2]
        return [tuple(signals), tuple(diss)]

    def get_config(self):
        # Todo: add regularizer for all
        config = {'beta_initializer': initializers.serialize(self.beta_initializer),
                  'beta_regularizer': regularizers.serialize(self.beta_regularizer),
                  'beta_constraint': constraints.serialize(self.beta_constraint),
                  'norm_axis': self.norm_axis}
        super_config = super(GibbsRouting, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class Routing(Module):
    def __init__(self,
                 probability_transformation='flipmax',
                 **kwargs):

        self.probability_transformation = prob_trans.get(probability_transformation)

        # be sure to call this somewhere
        super(Routing, self).__init__(module_input=True,
                                      module_output=True,
                                      need_capsule_params=False,
                                      **kwargs)

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("The number of capsules must be equal to the number of prototypes. Necessary "
                                 "assumption for Routing. You provide " + str(input_shape[0][1]) + "!="
                                 + str(input_shape[1][1]) + ". Maybe you forgot the calling of a measuring module.")

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        signals = inputs[0]
        diss = inputs[1]

        # reshape of signal to vector necessary? signal.shape: (batch, proto_num, channels, caps_dim1, ..., caps_dimN)
        if self.input_spec[0].ndim > 4:
            signal_shape = mixed_shape(signals)
            signals = K.reshape(signals, signal_shape[0:3] + (-1,))

        with K.name_scope('routing_probabilities'):
            coefficients = self.probability_transformation(diss)

        with K.name_scope('signal_routing'):
            signals = K.batch_dot(coefficients, signals, [2, 2])

        with K.name_scope('dissimilarity_routing'):
            diss = K.squeeze(K.batch_dot(coefficients, K.expand_dims(diss, -1), [2, 2]), -1)

        # reshape to new output_vector_shape necessary?
        if self.input_spec[0].ndim > 4:
            signals = K.reshape(signals, signal_shape[0:2] + signal_shape[3:])

        return {0: signals, 1: diss}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        del signals[2]
        del diss[2]
        return [tuple(signals), tuple(diss)]

    def get_config(self):
        config = {'probability_transformation': prob_trans.serialize(self.probability_transformation)}
        super_config = super(Routing, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class SqueezeRouting(Module):
    def __init__(self, **kwargs):
        # be sure to call this somewhere
        super(SqueezeRouting, self).__init__(module_input=True,
                                             module_output=True,
                                             need_capsule_params=False,
                                             **kwargs)

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("The number of capsules must be equal to the number of prototypes. Necessary "
                                 "assumption for Routing. You provide " + str(input_shape[0][1]) + "!="
                                 + str(input_shape[1][1]) + ". Maybe you forgot the calling of a measuring module.")

            if input_shape[0][2] != 1:
                raise ValueError("The channel dimension must be one for squeezing. You provide: "
                                 + str(input_shape[0][2]))

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        signals = K.squeeze(inputs[0], 2)
        diss = K.squeeze(inputs[1], 2)

        return {0: signals, 1: diss}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        del signals[2]
        del diss[2]
        return [tuple(signals), tuple(diss)]
