# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import initializers
from keras.engine.topology import InputSpec
from keras import backend as K

from ..capsule import Module
from .. import probability_transformations as prob_trans
from .. import constraints, regularizers
from ..utils import linalg_funcs as linalg
from ..utils.caps_utils import mixed_shape
from ..utils import normalization_funcs as norm_funcs


# Todo: routing tests
# DynamicRouting and the other Modules can't not be mixed within one capsule. Because for diss means close to zero high
# prob whereas for Dynamic Routing means high value high prob. Further, for DynamicRouting the capsule should always
# have an own dissimilarity initializer
class DynamicRouting(Module):
    def __init__(self,
                 iterations=3,
                 norm_axis='capsules',
                 **kwargs):

        if iterations < 1:
            raise ValueError("The number of iterations must be greater zero.")
        self.iterations = iterations

        if norm_axis not in ('capsules', 'channels', 1, 2, -1):
            raise ValueError("norm_axis must be 'capsules' or 'channels'. You provide: " + str(norm_axis))
        if norm_axis == 'capsules':
            self.norm_axis = 1
        elif norm_axis == 'channels':
            self.norm_axis = 2
        else:
            self.norm_axis = norm_axis

        # be sure to call this somewhere
        super(DynamicRouting, self).__init__(module_input=True,
                                             module_output=False,
                                             need_capsule_params=False,
                                             **kwargs)

    def _build(self, input_shape):
        if input_shape[0][1] != input_shape[1][1]:
            raise ValueError("The number of capsules must be equal to the number of prototypes. Necessary assumption "
                             "for Dynamic Routing. You provide " + str(input_shape[0][1]) + "!="
                             + str(input_shape[1][1]))

        self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                           InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        signals = inputs[0]
        sim = inputs[1]

        # reshape of signal to vector necessary? signal.shape: (batch, proto_num, channels, caps_dim1, ..., caps_dimN)
        if self.input_spec[0].ndim > 4:
            signal_shape = mixed_shape(signals)
            signals = K.reshape(signals, signal_shape[0:3] + (-1,))

        # we return the length as trust value
        for i in range(self.iterations):
            with K.name_scope('routing'):
                with K.name_scope('routing_probabilities'):
                    coefficients = prob_trans.softmax(sim, axis=self.norm_axis, max_stabilization=True)

                with K.name_scope('signal_routing'):
                    outputs = K.batch_dot(coefficients, inputs[0], [2, 2])
                    outputs = norm_funcs.dynamic_routing_squash(outputs)

                with K.name_scope('similarity_routing'):
                    if i < self.iterations - 1:
                        sim += K.batch_dot(outputs, signals, [2, 3])
                    else:
                        sim = K.squeeze(K.batch_dot(coefficients, K.expand_dims(sim, -1), [2, 2]), -1)

        with K.name_scope('get_probabilities'):
            # we have to place a really small epsilon. Otherwise the network learns nothing
            prob = linalg.norm(outputs, epsilon=1.e-14)

        # reshape to new output_vector_shape necessary?
        if self.input_spec[0].ndim > 4:
            outputs = K.reshape(outputs, signal_shape[0:2] + signal_shape[3:])

        return {0: outputs, 1: sim, 2: prob}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        del signals[2]
        del diss[2]
        return [tuple(signals), tuple(diss), tuple(diss)]

    def get_config(self):
        config = {'iterations': self.iterations,
                  'norm_axis': self.norm_axis}
        super_config = super(DynamicRouting, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


# Todo: Test of softmax with scaling
class GibbsClassification(Module):
    def __init__(self,
                 beta_initializer='ones',
                 beta_regularizer=None,
                 beta_constraint='NonNeg',
                 **kwargs):

        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.beta = None

        # be sure to call this somewhere
        super(GibbsClassification, self).__init__(module_input=True,
                                                  module_output=False,
                                                  need_capsule_params=True,
                                                  **kwargs)

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("The number of capsules must be the same in diss and signals. You provide "
                                 + str(input_shape[0][1]) + "!=" + str(input_shape[1][1]) + ". Maybe you forgot the "
                                 "calling of a routing/ competition module.")

            if input_shape[0][1] != self.capsule_number:
                raise ValueError("The defined number of capsules is not equal the number of capsules in signals. " +
                                 "You provide: " + str(input_shape[0][1]) + "!=" + str(self.capsule_number) +
                                 ". Maybe you forgot the calling of a competition module.")

            self.beta = self.add_weight(shape=(1,),
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        name='beta')

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        signals = inputs[0]
        diss = inputs[1]

        with K.name_scope('get_probabilities'):
            prob = prob_trans.neg_softmax(diss * self.beta, axis=-1, max_stabilization=True)

        return {0: signals, 1: diss, 2: prob}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        return [tuple(signals), tuple(diss), tuple(diss)]

    def get_config(self):
        config = {'beta_initializer': initializers.serialize(self.beta_initializer),
                  'beta_regularizer': regularizers.serialize(self.beta_regularizer),
                  'beta_constraint': constraints.serialize(self.beta_constraint)}
        super_config = super(GibbsClassification, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class Classification(Module):
    def __init__(self,
                 probability_transformation='neg_softmax',
                 **kwargs):

        self.probability_transformation = prob_trans.get(probability_transformation)

        # be sure to call this somewhere
        super(Classification, self).__init__(module_input=True,
                                             module_output=False,
                                             need_capsule_params=True,
                                             **kwargs)

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("The number of capsules must be the same in diss and signals. You provide "
                                 + str(input_shape[0][1]) + "!=" + str(input_shape[1][1]) + ". Maybe you forgot the "
                                 "calling of a routing/ competition module.")

            if input_shape[0][1] != self.capsule_number:
                raise ValueError("The defined number of capsules is not equal the number of capsules in signals. " +
                                 "You provide: " + str(input_shape[0][1]) + "!=" + str(self.capsule_number) +
                                 ". Maybe you forgot the calling of a competition module.")

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        signals = inputs[0]
        diss = inputs[1]

        with K.name_scope('get_probabilities'):
            prob = self.probability_transformation(diss)

        return {0: signals, 1: diss, 2: prob}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        return [tuple(signals), tuple(diss), tuple(diss)]

    def get_config(self):
        config = {'probability_transformation': prob_trans.serialize(self.probability_transformation)}
        super_config = super(Classification, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class Regression(Module):
    def __init__(self,
                 probability_transformation='neg_exp',
                 **kwargs):

        self.probability_transformation = prob_trans.get(probability_transformation)

        # be sure to call this somewhere
        super(Regression, self).__init__(module_input=True,
                                         module_output=False,
                                         need_capsule_params=True,
                                         **kwargs)

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("The number of capsules must be the same in diss and signals. You provide "
                                 + str(input_shape[0][1]) + "!=" + str(input_shape[1][1]) + ". Maybe you forgot the "
                                 "calling of a routing/ competition module.")

            if input_shape[0][1] != self.capsule_number:
                raise ValueError("The defined number of capsules is not equal the number of capsules in signals. " +
                                 "You provide: " + str(input_shape[0][1]) + "!=" + str(self.capsule_number) +
                                 ". Maybe you forgot the calling of a competition module.")

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        signals = inputs[0]
        diss = inputs[1]

        with K.name_scope('get_probabilities'):
            prob = self.probability_transformation(diss)

        return {0: signals, 1: diss, 2: prob}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        return [tuple(signals), tuple(diss), tuple(diss)]

    def get_config(self):
        config = {'probability_transformation': prob_trans.serialize(self.probability_transformation)}
        super_config = super(Regression, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))
